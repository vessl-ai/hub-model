# Segmind Vega

Distilled Stable diffusion XL while preserving high-quality text-to-image generation with 70% smaller size and 100% faster inference latency

## Description

The Segmind Vega Model is a distilled version of the Stable Diffusion XL (SDXL). The UNet in Segmind Vega has 745M parameters, which is 30% of the number of parameters of UNet in SDXL. Thus, the speed of inference is boasted up to twice of SDXL model while maintaining the high-quality generation.

The Segmind VegaRT is a distilled LCM-LoRA adapter for the Vega model. VegaRT generates images faster by reducing inference steps to 2-8 steps, so that it takes under 0.1 seconds on an Nvidia A100 GPU.

## Quickstart

```sh
# clone repository and move to directory
git clone https://github.com/vessl-ai/hub-model
cd segmind-vega

# run Vega on local
streamlit run app.py --server.port=80

# run VegaRT on local
streamlit run app_rt.py --server.port=80
```

## Reference

- Blog: <https://blog.segmind.com/segmind-vega/>
