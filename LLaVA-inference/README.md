# LLaVA

Visual instruction tuning towards large language and vision models with GPT-4 level capabilities

## Description

LLaVA is an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding. It combines an image feature extracted by CLIP and a language instruction embedding and processes them with an LLM model such as Vicuna or Llama. As a result, LLaVA has a visual chat capability and visual reasoning.

## Quickstart

```sh
pip install -r requirments.txt
git clone https://github.com/haotian-liu/LLaVA.git LLaVA-git
cd LLaVA-git
pip install -e .
cd ..
streamlit run streamlit_app.py --server.port 80
```

## Reference

- Arxiv: <https://arxiv.org/abs/2304.08485>
- Arxiv (v1.5): <https://arxiv.org/abs/2310.03744>
- Repository: <https://github.com/haotian-liu/LLaVA>
