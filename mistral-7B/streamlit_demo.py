from pathlib import Path
import subprocess

import streamlit as st
from PIL import Image

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from main import generate

tokenizer = Tokenizer("/model/mistral-7B-v0.1/tokenizer.model")
transformer = Transformer.from_folder(Path("/model/mistral-7B-v0.1/"), max_batch_size=1)

st.set_page_config(layout="wide")

image_path = "./assets/mistral-7B-v0.1.jpg"

col1, col2 = st.columns(2)

with col1:
    input_img = Image.open(image_path)
    st.image(input_img)
with col2:
    input_query = st.text_input("Input Query", "Tom sees flowers")
    temperature = st.slider("Temperature", 0.00, 1.00, 0.70, 0.01)

    if st.button("Generate"):
        res, _logprobs = generate(
            [input_query],
            transformer,
            tokenizer,
            max_tokens=256,
            temperature=temperature,
        )
        output_sentences = res[0]
        print(output_sentences)
        st.write(output_sentences)