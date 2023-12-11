# SSD-1B

SSD-1B: Faster, smaller SDXL variant excelling in high-quality text-to-image generation with diverse dataset training, incorporating Grit and Midjourney data.

## Description

The Segmind Stable Diffusion Model (`SSD-1B`) is a distilled **50% smaller version** of the Stable Diffusion XL (SDXL), offering a **60% speedup** while maintaining high-quality text-to-image generation capabilities. It has been trained on diverse datasets, including Grit and Midjourney scrape data, to enhance its ability to create a wide range of visual content based on textual prompts.

## Quickstart

Recreate the model's outcome using a concise code snippet (specifics of the YAML code can be found below).

```sh
# clone repository and move to directory
git clone https://github.com/vessl-ai/hub-model
cd SSD-1B

# run by cli
python ssd_1b_inference.py

# run streamlit on local
streamlit run ssd_1b_streamlit.py --server.port=80
```

## Reference

- Blog: <https://blog.segmind.com/introducing-segmind-ssd-1b/>
