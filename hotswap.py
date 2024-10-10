# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from operator import attrgetter

import torch

from peft.config import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING


from peft.utils.other import infer_device
from peft.utils.peft_types import PeftType
from peft.utils.save_and_load import load_peft_weights

PEFT_TYPE_TO_PREFIX_MAPPING = {
    PeftType.IA3: "ia3_",
    PeftType.LORA: "lora_",
    PeftType.ADALORA: "lora_",
    PeftType.LOHA: "hada_",
    PeftType.LOKR: "lokr_",
    PeftType.OFT: "oft_",
    PeftType.POLY: "poly_",
    PeftType.BOFT: "boft_",
    PeftType.LN_TUNING: "ln_tuning_",
    PeftType.VERA: "vera_lambda_",
    PeftType.FOURIERFT: "fourierft_",
    PeftType.HRA: "hra_",
    PeftType.VBLORA: "vblora_",
}

def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Utility function to remap the state_dict keys to fit the PEFT model by inserting the adapter name."""
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def hotswap_adapter_from_state_dict(model, state_dict, adapter_name, parameter_prefix="lora_"):
    """
    Swap out the adapter weights from the model with the weights from state_dict.
    As of now, only LoRA is supported.
    This is a low-level function that assumes that the adapters have been checked for compatibility and that the
    state_dict has been correctly mapped to work with PEFT. For a high level function that performs this work for you,
    use `hotswap_adapter` instead.
    Args:
        model (`nn.Module`):
            The model with the loaded adapter.
        state_dict (`dict[str, torch.Tensor]`):
            The state dict of the new adapter, which needs to be compatible (targeting same modules etc.).
        adapter_name (`str`):
            The name of the adapter that should be hot-swapped, e.g. `"default"`. The name will remain the same after
            swapping.
        parameter_prefix (`str`, *optional*, defaults to `"lora_"`)
            The prefix used to identify the adapter's keys in the state dict. For LoRA, this would be `"lora_"` (the
            default).
    Raises:
        RuntimeError
            If the old and the new adapter are not compatible, a RuntimeError is raised.
    """
    # Ensure that all the keys of the new adapter correspond exactly to the keys of the old adapter, otherwise
    # hot-swapping is not possible

    is_compiled = hasattr(model, "_orig_mod")
    # TODO: there is probably a more precise way to identify the adapter keys
    missing_keys = {k for k in model.state_dict() if (parameter_prefix in k) and (adapter_name in k)}
    unexpected_keys = set()

    # first: dry run, not swapping anything
    for key, new_val in state_dict.items():
        key = key.removeprefix("transformer.") # TODO: this is but one of many state dict hacks; we'll need to establish some sort of pattern to remap here
        try:
            old_val = attrgetter(key)(model)
        except AttributeError:
            unexpected_keys.add(key)
            continue

        if is_compiled:
            missing_keys.remove("_orig_mod." + key)
        else:
            missing_keys.remove(key)

    if missing_keys or unexpected_keys:
        msg = "Hot swapping the adapter did not succeed."
        if missing_keys:
            msg += f" Missing keys: {', '.join(sorted(missing_keys))}."
        if unexpected_keys:
            msg += f" Unexpected keys: {', '.join(sorted(unexpected_keys))}."
        raise RuntimeError(msg)

    # actual swapping
    for key, new_val in state_dict.items():
        key = key.removeprefix("transformer.") # TODO: see above
        # no need to account for potential _orig_mod in key here, as torch handles that
        old_val = attrgetter(key)(model)
        if is_compiled:
            # here we zero pad
            old_val.data = new_val.data
        else:
            torch.utils.swap_tensors(old_val, new_val)


def _check_hotswap_configs_compatible(config0: PeftConfig, config1: PeftConfig) -> None:
    """
    Check if two configs are compatible for hot-swapping.
    Only LoRA parameters are checked for now.
    To hot-swap two adapters, their configs must be compatible. Otherwise, the results could be false. E.g. if they use
    different alpha values, after hot-swapping, the alphas from the first adapter would still be used with the weights
    from the 2nd adapter, which would result in incorrect behavior. There is probably a way to swap these values as
    well, but that's not implemented yet, and we need to be careful not to trigger re-compilation if the model is
    compiled (so no modification of the dict).
    """

    if config0.peft_type != config1.peft_type:
        msg = f"Incompatible PEFT types found: {config0.peft_type.value} and {config1.peft_type.value}"
        raise ValueError(msg)

    # TODO: This is a very rough check only for LoRA at the moment. Also, there might be some options that don't
    # necessarily require an error.
    config_keys_to_check = ["lora_alpha", "use_rslora", "lora_dropout", "alpha_pattern", "use_dora"]
    config0 = config0.to_dict()
    config1 = config1.to_dict()
    sentinel = object()
    for key in config_keys_to_check:
        val0 = config0.get(key, sentinel)
        val1 = config1.get(key, sentinel)
        if val0 != val1:
            raise ValueError(f"Configs are incompatible: for {key}, {val0} != {val1}")


def hotswap_adapter(pipe, model_name_or_path, adapter_name, torch_device="cuda", **kwargs):
    """Substitute old adapter data with new adapter data, keeping the rest the same.
    As of now, only LoRA is supported.
    This function is useful when you want to replace the loaded adapter with a new adapter. The adapter name will
    remain the same, but the weights and other parameters will be swapped out.
    If the adapters are incomptabile, e.g. targeting different layers or having different alpha values, an error will
    be raised.
    Args:
        model ([`~PeftModel`]):
            The PEFT model with the loaded adapter.
        model_name_or_path (`str`):
            The name or path of the model to load the new adapter from.
        adapter_name (`str`):
            The name of the adapter to swap, e.g. `"default"`. The name will stay the same after swapping.
        torch_device: (`str`, *optional*, defaults to None):
            The device to load the new adapter onto.
        **kwargs (`optional`):
            Additional keyword arguments used for loading the config and weights.
    """

    ############################

    state_dict, alphas = pipe.lora_state_dict(model_name_or_path, return_alphas=True)
    if alphas:
        print("ignoring alphas! not that hard to unignore but that is a problem for The Future")

    ###########################
    # LOAD & REMAP STATE_DICT #
    ###########################

    parameter_prefix = "lora_"
    peft_model_state_dict = _insert_adapter_name_into_state_dict(
        state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
    )

    hotswap_adapter_from_state_dict(
        model=pipe.transformer,
        state_dict=peft_model_state_dict,
        adapter_name=adapter_name,
        parameter_prefix=parameter_prefix,
    )
