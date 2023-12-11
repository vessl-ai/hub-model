# Finetuning Llama2

Finetuning Llama 2 with a small code instruction dataset in Alpaca format

## Description

Llama 2 is a large language model (LLM) developed by Meta. It is open-sourced with codes and various weights pre-trained and finetuned on, varying in size of parameter counts from 7 billion to 70 billion. In this repository, llama 2 7B model will be finetuned on a code instruction dataset which consists of 18k samples and is in format of Alpaca instruction. 

While training, learning rate, training loss and validation loss will be logged on `Plots` page.

## Quickstart

Modify `config.yaml` to your own configuration and run following:

```sh
pip install -r requirements.txt
python finetuning.py
```

## Reference

- Arxiv: <https://arxiv.org/abs/2307.09288>
- Blog: <https://ai.meta.com/resources/models-and-libraries/llama/>
- Repository: <https://github.com/facebookresearch/llama>
- Original Dataset: <https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca>
