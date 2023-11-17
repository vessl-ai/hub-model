# SSD-1B

SSD-1B: Faster, smaller SDXL variant excelling in high-quality text-to-image generation with diverse dataset training, incorporating Grit and Midjourney data.

# Description

The Segmind Stable Diffusion Model (`SSD-1B`) is a distilled **50% smaller version** of the Stable Diffusion XL (SDXL), offering a **60% speedup** while maintaining high-quality text-to-image generation capabilities. It has been trained on diverse datasets, including Grit and Midjourney scrape data, to enhance its ability to create a wide range of visual content based on textual prompts.

# Quickstart

Recreate the model's outcome using a concise code snippet (specifics of the YAML code can be found below).

```sh
# clone repository and move to directory
git clone https://github.com/vessl-ai/hub-model
cd SSD-1B

# run inference via vessl run
vessl run create -f SSD-1B.yaml
```

Afterward, accessing the Streamlit port will show you:

[attach image here]

## YAML for `vessl run` (enable infernce & streamlit together)

Inference is done by `python SSD-1B_inference.py`

Streamlit is done by `streamlit run SSD-1B_streamlit.py --server.port=80`

```yaml
name: inference_SSD-1B_A10G
resources:
  cluster: aws-apne2
  preset: v1.a10g-1.mem-26
image: quay.io/vessl-ai/ngc-pytorch-kernel:23.09-py3-202310300302
import:
  /dataset/: vessl-model://vessl-ai/SSD-1B/1
  /root/examples/: git://github.com/vessl-ai/examples.git
run:
  - command: |-
      pip install --upgrade pip
      pip install -r requirements.txt
      pip install git+https://github.com/huggingface/diffusers
      mkdir /data
      cd /dataset
      mv SSD-1B.tar.gz /data
      cd /data/
      tar -xvf SSD-1B.tar.gz
      cd /root/examples/SSD-1B
      python SSD-1B_inference.py
      streamlit run SSD-1B_streamlit.py --server.port=80
    workdir: /root/examples/SSD-1B
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - name: streamlit
    type: http
    port: 80
```