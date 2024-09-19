# FLUX.1-dev LoRA Explorer Cog Model

This is an implementation of [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as a [Cog](https://github.com/replicate/cog) model.

CodeName LoRA Explorer, to explore the model with different LoRA weights.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="a beautiful castle frstingln illustration" -i extra_lora="alvdansen/frosting_lane_flux"

![output](output.png)


## Tests

HF
```
cog predict -i prompt="a beautiful castle frstingln illustration" -i hf_lora="alvdansen/frosting_lane_flux" -i output_format="png"
```

CivitAI
```
cog predict -i prompt="pnt style Illustration of a wizard" -i hf_lora="https://civitai.com/api/download/models/735262?type=Model&format=SafeTensor" -i output_format="png"
```

Replicate
```
cog predict -i prompt="photo of TOK with purple hair" -i hf_lora="https://replicate.delivery/yhqm/9vSmRCa8Vv7bFtKfCfXTRzTq4X71tZW0LtLCb1l49bTSo8TTA/trained_model.tar" -i output_format="png"
```

## License

The code in this repository is licensed under the [Apache-2.0 License](LICENSE).

Flux Dev falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

`FLUX.1 [dev]` fine-tuned weights and their outputs are non-commercial by default, but can be used commercially when running on Replicate.
