"""
🐶 Dog vs Cat 🐱 — Streamlit inference app
Model: CNN trained on 150x150 RGB images, sigmoid output (0 = Cat, 1 = Dog).
Run:   streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------- config
MODEL_PATH = os.environ.get("MODEL_PATH", "dogs_vs_cats_model.keras")
IMG_HEIGHT, IMG_WIDTH = 150, 150  # must match training
CLASS_NAMES = {0: "Cat", 1: "Dog"}  # flow_from_directory: cats=0, dogs=1
EMOJI = {"Cat": "🐱", "Dog": "🐶"}

st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐾", layout="centered")


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def preprocess(img: Image.Image) -> np.ndarray:
    """Same pipeline as the notebook: resize to 150x150, scale to [0, 1]."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)  # keras load_img default
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img: Image.Image):
    p_dog = float(model.predict(preprocess(img), verbose=0)[0][0])
    label = CLASS_NAMES[int(p_dog > 0.5)]
    confidence = p_dog if label == "Dog" else 1.0 - p_dog
    return label, confidence, p_dog


# ---------------------------------------------------------------- UI
st.title("🐶 Dog vs Cat 🐱")
st.caption("CNN image classifier — upload a photo (or use your camera) and get a prediction.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`. Place `dogs_vs_cats_model.keras` "
        "next to `app.py` or set the `MODEL_PATH` environment variable."
    )
    st.stop()

model = load_model(MODEL_PATH)

with st.sidebar:
    st.header("About")
    st.write(
        "Simple 4-block CNN (Conv → MaxPool ×4, Dense 512, Dropout 0.5) "
        f"trained on {IMG_WIDTH}×{IMG_HEIGHT} images."
    )
    threshold = st.slider("Decision threshold (P(Dog))", 0.05, 0.95, 0.50, 0.05)
    st.caption("Raise it to be stricter before calling something a dog.")
    show_raw = st.checkbox("Show raw model output", value=False)

tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])
with tab_upload:
    files = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )
with tab_camera:
    cam = st.camera_input("Take a photo")

images = [(f.name, f) for f in (files or [])]
if cam is not None:
    images.append(("camera.jpg", cam))

if not images:
    st.info("Upload an image to get started.")
    st.stop()

for name, f in images:
    try:
        img = Image.open(f)
        img.load()
    except Exception:
        st.warning(f"Could not read `{name}` as an image.")
        continue

    _, _, p_dog = predict(model, img)
    label = CLASS_NAMES[int(p_dog > threshold)]
    confidence = p_dog if label == "Dog" else 1.0 - p_dog

    st.divider()
    col_img, col_res = st.columns([1, 1], vertical_alignment="center")
    with col_img:
        st.image(img, caption=name, use_container_width=True)
    with col_res:
        st.subheader(f"{EMOJI[label]} {label}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(1.0 - p_dog, text=f"Cat {(1 - p_dog) * 100:.1f}%")
        st.progress(p_dog, text=f"Dog {p_dog * 100:.1f}%")
        if confidence < 0.65:
            st.caption("⚠️ Low confidence — the image may be unclear or not a cat/dog.")
        if show_raw:
            st.code(f"sigmoid output (P(Dog)) = {p_dog:.6f}")

st.divider()
st.caption(
    "Note: the model only knows cats and dogs — any other image will still be "
    "forced into one of those two classes."
)
