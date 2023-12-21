import io

import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from st_keyup import st_keyup

st.set_page_config(layout="wide")
st.title("Segmind VegaRT")

if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = io.BytesIO()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "/model/vega",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    pipe.load_lora_weights("/model/vegart")
    pipe.fuse_lora()

    return pipe


pipe = load_model()

col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Steps", 2, 8, 4, 1)

with col2:
    guidance_scale = st.slider("Guidance Scale", 0.0, 2.0, 0.0)

prompt = st_keyup("Prompt", debounce=500)

if prompt:
    with st.spinner("Generating image..."):
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ).images[0]

        image.save(st.session_state.image_bytes, format="PNG")
        st.image(st.session_state.image_bytes)
