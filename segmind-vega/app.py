import io

import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline

st.set_page_config(layout="wide")
st.title("Segmind Vega")
col1, col2 = st.columns([2, 3])

if "generate_flag" not in st.session_state:
    st.session_state["generate_flag"] = False

if "generated" not in st.session_state:
    st.session_state["generated"] = False

if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = io.BytesIO()


def turn_on_generate_flag():
    st.session_state["generate_flag"] = True


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "/model/vega",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda")

    return pipe


pipe = load_model()

with col1:
    input_query = "A cinematic shot of a racoon wearing an intricate italian priest robe, with a crown"  # Your prompt here
    prompt = st.text_area("Input Query", input_query)

    negative_query = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)"  # Negative prompt here
    negative_prompt = st.text_area("Negative Things", negative_query)

    steps = st.slider("Steps", 10, 100, 20, 1)
    guidance_scale = st.slider("Guidance Scale", 1.0, 25.0, 1.0, 0.1)

    button = st.button(
        "Generate",
        on_click=turn_on_generate_flag,
        type="primary",
        use_container_width=True,
    )

with col2:
    if st.session_state.generate_flag:
        progress_bar = st.progress(0.0, text="Running diffusion pipeline...")

        def update_progress(pipe, step_index, timestep, callback_kwargs):
            progress_bar.progress(
                (step_index + 1) / steps, text="Running diffusion pipeline..."
            )

            return callback_kwargs

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            callback_on_step_end=update_progress,
        ).images[0]
        progress_bar.empty()

        image.save(st.session_state.image_bytes, format="PNG")

        st.session_state.generate_flag = False
        st.session_state.generated = True

    if st.session_state.generated:
        st.session_state.image_bytes.seek(0)
        st.image(st.session_state.image_bytes)
