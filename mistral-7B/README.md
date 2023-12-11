# Mistral 7B

Mistral-7B: An open-source LLM which achieves both efficiency and high performance.

## Description

Mistral-7B is open-source LLM developed by Mistral AI. The model utilizes a grouped query attention (GQA) and a sliding window attention mechanism (SWA), which enable faster inference and handling longer sequences at smaller cost than other models. As a result, it achieves both efficiency and high performance. Mistral-7B outperforms Llama 2 13B on all benchmarks and Llama 1 34B in reasoning, mathematics, and code generation benchmarks.

## Quickstart

### interactive demo

```sh
pip install -r requirements_streamlit.txt

python -m main interactive /path/to/mistral-7B-v0.1/
```

### streamlit 

```sh
pip install -r requirements_streamlit.txt

streamlit run streamlit_demo.py --server.port=80
```

## Reference

- Arxiv: <https://arxiv.org/abs/2310.06825>
- Blog: <https://mistral.ai/news/announcing-mistral-7b/>
- Original Code: <https://github.com/mistralai/mistral-src>
