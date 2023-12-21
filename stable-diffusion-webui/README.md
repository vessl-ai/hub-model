# Stable Diffusion WebUI

Run stable diffusion with multiple checkpoints on web app

## Description

Stable diffusion is a deep learning, text-to-image model that uses a diffusion technique, which generates an image from noise by iterating gradual denoising steps. Unlike other previous text-to-image models, stable diffusion performs a diffusion process in the latent space with a smaller dimension and reconstructs the result to the image in real dimension. Also, cross attention mechanism is added for multi-modal tasks such as text-to-image and layout-to-image tasks.

In this example, a simple web app for stable diffusion inference is deployed. Some SD model checkpoints are mounted with vessl model, so that you can try some generation instantly.

## Quickstart

```sh
pip install -r requirements.txt

./webui.sh --ckpt-dir /path/to/your/ckpts
```

## Reference

- Arxiv: <https://arxiv.org/abs/2112.10752>
- Original Code: <https://github.com/AUTOMATIC1111/stable-diffusion-webui>
