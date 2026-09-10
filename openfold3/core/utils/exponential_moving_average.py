# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from torch._utils import _unflatten_dense_tensors

from openfold3.core.utils.tensor_utils import tensor_tree_map


class ExponentialMovingAverage:
    """
    Maintains moving averages of parameters with exponential decay

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`

    where `decay` is an attribute of the ExponentialMovingAverage object.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        submodules_to_update: list | None = None,
        offload_to_cpu: bool = False,
    ):
        """
        Args:
            model:
                A torch.nn.Module whose parameters are to be tracked
            decay:
                A value (usually close to 1.) by which updates are
                weighted as part of the above formula
            submodules_to_update:
                A list of submodules whose EMA weights will be updated.
                If not specified, all weights are updated.
            offload_to_cpu:
                Keep the shadow copy in host memory instead of on the
                accelerator. The shadow is only read at validation and
                checkpointing, and the update runs after the optimizer step
                when device memory is back at its floor, so this trades a
                per-step host transfer for a full parameter copy of device
                memory. Values are unchanged.
        """
        super().__init__()

        self.offload_to_cpu = offload_to_cpu

        self.params = {}
        model_params = model.state_dict()
        if "version_tensor" in model_params:
            self.params["version_tensor"] = (
                model_params["version_tensor"].detach().clone()
            )
        self.decay = decay
        self.submodules_to_update = submodules_to_update
        self.device = next(model.parameters()).device

    def init_params(self, model: torch.nn.Module):
        # SampleDiffusion registers the diffusion module as an alias, so
        # model.state_dict() exposes 763 of its tensors under two prefixes
        # each. Cloning per key would hold two EMA copies of the same weight.
        # Clone once per distinct source tensor and share it across aliases;
        # _update_state_dict_ skips repeats so the decay rate is unchanged.
        shared: dict[int, torch.Tensor] = {}

        def clone_param(t):
            if not isinstance(t, torch.Tensor):
                return t.detach().clone()

            ptr = t.data_ptr()
            if ptr not in shared:
                copy = t.detach().clone()
                if self.offload_to_cpu:
                    copy = copy.to("cpu")
                shared[ptr] = copy
            return shared[ptr]

        self.params = (
            self._init_flat_shadow(model.state_dict())
            if self.offload_to_cpu
            else tensor_tree_map(clone_param, model.state_dict())
        )
        self.device = next(model.parameters()).device

    def to(self, device):
        # When offloading, `device` is the compute device the shadow is being
        # tracked against, not where it lives; the callers only use it to
        # detect a device change, so record it without moving anything.
        if not self.offload_to_cpu:
            self.params = tensor_tree_map(lambda t: t.to(device), self.params)

        self.device = device
        return self

    def _update_state_dict_(self, update, state_dict, _updated=None):
        # Aliased state_dict keys share one stored tensor (see init_params), so
        # track what has already been decayed this call. Without this each
        # aliased weight would be decayed once per alias instead of once.
        if _updated is None:
            _updated = set()

        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored, _updated)
                else:
                    if id(stored) in _updated:
                        continue
                    _updated.add(id(stored))

                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def _init_flat_shadow(self, state_dict: dict) -> dict:
        """Build the host shadow as one pinned flat buffer per dtype.

        Entries of self.params become views into those buffers, so an update is
        a single host-to-device copy, fused device arithmetic, and a single copy
        back -- no per-tensor work anywhere. Aliased keys share a view, as in
        the on-device path.
        """
        first_key_for_storage: dict[int, str] = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                first_key_for_storage.setdefault(v.data_ptr(), k)

        distinct = list(first_key_for_storage.values())
        groups: dict[torch.dtype, list[str]] = {}
        for k in distinct:
            groups.setdefault(state_dict[k].dtype, []).append(k)

        self._flat_shadow = {}
        self._flat_keys = {}
        views: dict[str, torch.Tensor] = {}

        pin = torch.cuda.is_available()
        for dtype, keys in groups.items():
            sizes = [state_dict[k].numel() for k in keys]
            flat = torch.empty(sum(sizes), dtype=dtype, pin_memory=pin)
            offset = 0
            for k, numel in zip(keys, sizes, strict=True):
                view = flat.narrow(0, offset, numel).view(state_dict[k].shape)
                view.copy_(state_dict[k])
                views[k] = view
                offset += numel

            self._flat_shadow[dtype] = flat
            self._flat_keys[dtype] = keys

        # Point every aliased key at the view built for its storage
        return {
            k: views[first_key_for_storage[v.data_ptr()]]
            if isinstance(v, torch.Tensor)
            else v.detach().clone()
            for k, v in state_dict.items()
        }

    @torch.no_grad()
    def _update_flat_shadow_(self, update: dict) -> None:
        """Decay the host shadow toward `update`, computing on `update`'s device.

        The arithmetic is 368M elementwise ops per step; running it on the host
        instead measured 31-49% slower end to end, so the shadow makes a round
        trip and the maths stays on the accelerator. Op order matches
        _update_state_dict_ exactly: diff = stored - v; diff *= 1 - decay;
        stored -= diff.
        """
        for dtype, keys in self._flat_keys.items():
            present = [k for k in keys if isinstance(update.get(k), torch.Tensor)]
            if not present:
                continue

            flat_host = self._flat_shadow[dtype]
            device = update[present[0]].device
            flat_dev = flat_host.to(device, non_blocking=True)

            shadow_views = _unflatten_dense_tensors(
                flat_dev, [self.params[k] for k in keys]
            )
            by_key = dict(zip(keys, shadow_views, strict=True))

            stored = [by_key[k] for k in present]
            incoming = [update[k] for k in present]

            diffs = torch._foreach_sub(stored, incoming)
            torch._foreach_mul_(diffs, 1 - self.decay)
            torch._foreach_sub_(stored, diffs)

            flat_host.copy_(flat_dev)

    def update(self, model: torch.nn.Module) -> None:
        """
        Updates the stored parameters using the state dict of the provided
        module. The module should have the same structure as that used to
        initialize the ExponentialMovingAverage object.
        """
        if self.submodules_to_update is None:
            # If no subset is specified, update all parameters.
            if self.offload_to_cpu:
                self._update_flat_shadow_(model.state_dict())
            else:
                self._update_state_dict_(model.state_dict(), self.params)
            return

        # If a subset is specified, filter the state_dict to only include
        # parameters from the enabled submodules.
        update_dict = OrderedDict()
        model_state_dict = model.state_dict()

        for key, value in model_state_dict.items():
            is_enabled = any(
                key == prefix or key.startswith(f"{prefix}.")
                for prefix in self.submodules_to_update
            )
            if is_enabled:
                update_dict[key] = value

        if not update_dict:
            warnings.warn(
                "ExponentialMovingAverage: No parameters found for the specified "
                f"submodules_to_update: {self.submodules_to_update}.",
                stacklevel=2,
            )
            return

        if self.offload_to_cpu:
            self._update_flat_shadow_(update_dict)
        else:
            self._update_state_dict_(update_dict, self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        # Preserve the alias sharing established in init_params, so resuming
        # from a checkpoint does not silently reinflate the EMA copy.
        if self.offload_to_cpu:
            # params are views into the flat buffers; copy in place so the
            # buffers (and the alias sharing) survive a resume
            for k, v in state_dict["params"].items():
                self.params[k].copy_(v)
            self.decay = state_dict["decay"]
            return

        shared: dict[int, torch.Tensor] = {}
        for k, v in state_dict["params"].items():
            ptr = v.data_ptr()
            if ptr not in shared:
                shared[ptr] = v.clone()
            self.params[k] = shared[ptr]
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )
