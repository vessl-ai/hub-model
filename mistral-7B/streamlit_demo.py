import subprocess

import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

image_path = "./assets/mistral-7B-v0.1.jpg"

col1, col2 = st.columns(2)

with col1:
    input_img = Image.open(image_path)
    st.image(input_img)
with col2:
    input_query = st.text_input("Input Query", "Tom see flowers")
    temperature = st.slider("Temperature", 0.00, 1.00, 0.70, 0.01)

    cmd = f"python main.py single-inference /root/ckpt/mistral-7B-v0.1/ --max_tokens 256 --temperature {temperature} --prompt".split(
        " "
    )
    cmd.append(input_query)

    if st.button("Generate"):
        print(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        output_sentences = result.stdout.decode("utf-8")
        print(output_sentences)
        st.write(output_sentences)
