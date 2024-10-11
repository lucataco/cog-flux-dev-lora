# ruff: noqa
from typing import Dict, Union
import torch 
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
    get_peft_kwargs,
    is_peft_version,
    get_adapter_name,
    logging,
)

from hotswap import rank_up_state_dict

logger = logging.get_logger(__name__)


def lora_state_dict(
    cls,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    return_alphas: bool = False,
    pad_rank: int = None,
    **kwargs,
):
    r"""
    Return state dict for lora weights and the network alphas.

    <Tip warning={true}>

    We support loading A1111 formatted LoRA checkpoints in a limited capacity.

    This function is experimental and might change in the future.

    </Tip>

    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
            Can be either:

                - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                    the Hub.
                - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                    with [`ModelMixin.save_pretrained`].
                - A [torch state
                    dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.

        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        subfolder (`str`, *optional*, defaults to `""`):
            The subfolder location of a model file within a larger model repository on the Hub or locally.

    """
    # Load the main state dict first which has the LoRA layers for either of
    # transformer and text encoder or both.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)

    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    state_dict = cls._fetch_state_dict(
        pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
        weight_name=weight_name,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        subfolder=subfolder,
        user_agent=user_agent,
        allow_pickle=allow_pickle,
    )
    is_dora_scale_present = any("dora_scale" in k for k in state_dict)
    if is_dora_scale_present:
        warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
        logger.warning(warn_msg)
        state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

    # TODO (sayakpaul): to a follow-up to clean and try to unify the conditions.
    is_kohya = any(".lora_down.weight" in k for k in state_dict)
    if is_kohya:
        state_dict = _convert_kohya_flux_lora_to_diffusers(state_dict)
        # Kohya already takes care of scaling the LoRA parameters with alpha.
        return (state_dict, None) if return_alphas else state_dict

    is_xlabs = any("processor" in k for k in state_dict)
    if is_xlabs:
        state_dict = _convert_xlabs_flux_lora_to_diffusers(state_dict)
        # xlabs doesn't use `alpha`.
        return (state_dict, None) if return_alphas else state_dict

    # For state dicts like
    # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA
    keys = list(state_dict.keys())
    network_alphas = {}
    for k in keys:
        if "alpha" in k:
            alpha_value = state_dict.get(k)
            if (torch.is_tensor(alpha_value) and torch.is_floating_point(alpha_value)) or isinstance(
                alpha_value, float
            ):
                network_alphas[k] = state_dict.pop(k)
            else:
                raise ValueError(
                    f"The alpha key ({k}) seems to be incorrect. If you think this error is unexpected, please open as issue."
                )
    if pad_rank: 

        state_dict = rank_up_state_dict(state_dict, pad_rank)


    if return_alphas:
        return state_dict, network_alphas
    else:
        return state_dict

# patching inject_adapter_in_model and load_peft_state_dict with low_cpu_mem_usage=True until it's merged into diffusers
def load_lora_into_transformer(
    cls, state_dict, network_alphas, transformer, adapter_name=None, _pipeline=None
):
    """
    This will load the LoRA layers specified in `state_dict` into `transformer`.

    Parameters:
        state_dict (`dict`):
            A standard state dict containing the lora layer parameters. The keys can either be indexed directly
            into the unet or prefixed with an additional `unet` which can be used to distinguish between text
            encoder lora layers.
        network_alphas (`Dict[str, float]`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the
            same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
            link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
        transformer (`SD3Transformer2DModel`):
            The Transformer model to load the LoRA layers into.
        adapter_name (`str`, *optional*):
            Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
            `default_{i}` where i is the total number of adapters being loaded.
    """
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    keys = list(state_dict.keys())

    transformer_keys = [k for k in keys if k.startswith(cls.transformer_name)]
    state_dict = {
        k.replace(f"{cls.transformer_name}.", ""): v
        for k, v in state_dict.items()
        if k in transformer_keys
    }

    if len(state_dict.keys()) > 0:
        # check with first key if is not in peft format
        first_key = next(iter(state_dict.keys()))
        if "lora_A" not in first_key:
            state_dict = convert_unet_state_dict_to_peft(state_dict)

        if adapter_name in getattr(transformer, "peft_config", {}):
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
            )

        rank = {}
        for key, val in state_dict.items():
            if "lora_B" in key:
                rank[key] = val.shape[1]

        if network_alphas is not None and len(network_alphas) >= 1:
            prefix = cls.transformer_name
            alpha_keys = [
                k
                for k in network_alphas.keys()
                if k.startswith(prefix) and k.split(".")[0] == prefix
            ]
            network_alphas = {
                k.replace(f"{prefix}.", ""): v
                for k, v in network_alphas.items()
                if k in alpha_keys
            }

        lora_config_kwargs = get_peft_kwargs(
            rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict
        )
        if "use_dora" in lora_config_kwargs:
            if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                )
            else:
                lora_config_kwargs.pop("use_dora")
        lora_config = LoraConfig(**lora_config_kwargs)

        # adapter_name
        if adapter_name is None:
            adapter_name = get_adapter_name(transformer)

        # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
        # otherwise loading LoRA weights will lead to an error
        is_model_cpu_offload, is_sequential_cpu_offload = (
            cls._optionally_disable_offloading(_pipeline)
        )

        inject_adapter_in_model(
            lora_config, transformer, adapter_name=adapter_name, low_cpu_mem_usage=True
        )
        incompatible_keys = set_peft_model_state_dict(
            transformer, state_dict, adapter_name, low_cpu_mem_usage=True
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />