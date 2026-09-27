"""
Leaf Diagnostic - Streamlit Community Cloud deployment version
------------------------------------------------------------
Free, no-card deployment: share.streamlit.io connects directly to a
public GitHub repo. Put this file, requirements.txt, best_model.keras
and class_names.txt together at the root of that repo.
"""

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib
import matplotlib.cm as cm

MODEL_PATH = "best_model.keras"
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "out_relu"  # MobileNetV2's final conv activation layer

st.set_page_config(page_title="Leaf Diagnostic", page_icon="🌿", layout="wide")


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        names = [line.strip() for line in f if line.strip()]

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        raise RuntimeError(
            "Could not find the MobileNetV2 backbone inside the loaded model."
        )

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(LAST_CONV_LAYER_NAME).output,
    )

    head_layers = []
    after_backbone = False
    for layer in model.layers:
        if layer is base_model:
            after_backbone = True
            continue
        if after_backbone:
            head_layers.append(layer)

    return model, names, grad_model, head_layers


model, class_names, grad_model, head_layers = load_assets()


def run_head(x):
    for layer in head_layers:
        x = layer(x, training=False)
    return x


def load_and_resize(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def make_gradcam_overlay(raw_batch, predicted_index, original_pil):
    backbone_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        raw_batch.copy()
    )

    with tf.GradientTape() as tape:
        conv_output = grad_model(backbone_input)
        tape.watch(conv_output)
        preds = run_head(conv_output)
        class_channel = preds[:, predicted_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_arr = np.array(heatmap_img)

    try:
        colormap = matplotlib.colormaps["jet"]
    except AttributeError:
        colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_arr / 255.0)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    original_resized = np.array(original_pil.convert("RGB").resize(IMG_SIZE))
    overlay = np.uint8(original_resized * 0.6 + colored_heatmap * 0.4)
    return Image.fromarray(overlay)


def confidence_bar_html(label, pct, highlight):
    color = "var(--color-leaf)" if highlight else "var(--color-soil-light)"
    return f"""
    <div class="bar-row">
      <div class="bar-label">{label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
      <div class="bar-pct">{pct:.1f}%</div>
    </div>
    """


def build_results_html(predicted_class, confidence, top3):
    is_healthy = "healthy" in predicted_class.lower()
    verdict_color = "var(--color-leaf)" if is_healthy else "var(--color-ochre)"
    verdict_word = "Healthy" if is_healthy else "Disease detected"

    bars = "".join(
        confidence_bar_html(name, pct, highlight=(name == predicted_class))
        for name, pct in top3
    )

    return f"""
    <div class="readout">
      <div class="readout-top">
        <span class="verdict-dot" style="background:{verdict_color};"></span>
        <span class="verdict-word">{verdict_word}</span>
      </div>
      <div class="readout-class">{predicted_class}</div>
      <div class="readout-confidence">{confidence:.2f}<span class="unit">% confidence</span></div>
      <div class="readout-divider"></div>
      <div class="bars">{bars}</div>
    </div>
    """


def predict_and_explain(pil_img):
    try:
        raw_batch = load_and_resize(pil_img)
        predictions = model.predict(raw_batch, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        predicted_class = class_names[predicted_index]

        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = [(class_names[i], float(predictions[0][i] * 100)) for i in top3_idx]

        overlay_img = make_gradcam_overlay(raw_batch, predicted_index, pil_img)
        html = build_results_html(predicted_class, confidence, top3)
        return html, overlay_img

    except Exception as e:
        return f"<div class='readout'>Something went wrong: {e}</div>", None


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
FONT_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
"""

CUSTOM_CSS = """
:root {
  --color-canopy: #1F3D2B;
  --color-canopy-light: #2C4A34;
  --color-parchment: #EEF1E6;
  --color-soil: #6B4A2F;
  --color-soil-light: #B7A488;
  --color-ochre: #B5762A;
  --color-leaf: #4C7A3D;
  --color-ink: #201F1A;
}

.stApp { background: var(--color-parchment) !important; color: var(--color-ink) !important;
  font-family: 'IBM Plex Sans', sans-serif !important; }

.hero { background: var(--color-canopy); color: var(--color-parchment); padding: 26px 30px;
  border-radius: 6px; margin-bottom: 18px; }
.hero h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; margin: 0 0 8px 0; }
.hero p { margin: 0; max-width: 62ch; color: #D8E0D0; line-height: 1.5; }

.readout-top { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.verdict-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.verdict-word { font-family:'IBM Plex Mono', monospace; font-size:0.85rem; color: var(--color-soil); }
.readout-class { font-family:'IBM Plex Mono', monospace; font-size:1.35rem; font-weight:600; margin:2px 0 6px 0; }
.readout-confidence { font-size:2.1rem; font-family:'IBM Plex Mono', monospace; color: var(--color-ink); margin-bottom:12px; }
.readout-confidence .unit { font-size:0.85rem; font-family:'IBM Plex Sans', sans-serif; color: var(--color-soil); margin-left:6px; }
.readout-divider { height:1px; background: var(--color-soil-light); margin: 8px 0 12px 0; }

.bar-row { display:flex; align-items:center; gap:10px; margin-bottom:7px; font-size:0.82rem; }
.bar-label { width:150px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-track { flex:1; height:8px; background:#E4E1D3; border-radius:2px; overflow:hidden; }
.bar-fill { height:100%; }
.bar-pct { width:46px; text-align:right; font-family:'IBM Plex Mono', monospace; }

.stButton > button { background: var(--color-canopy) !important; border-color: var(--color-canopy) !important;
  color: var(--color-parchment) !important; font-family: 'IBM Plex Mono', monospace !important; }
.stButton > button:hover { background: var(--color-canopy-light) !important; color: var(--color-parchment) !important; }

.disclaimer { font-size:0.8rem; color: var(--color-soil); margin-top:14px; max-width:75ch; }
"""

st.markdown(FONT_HTML + f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>Leaf Diagnostic</h1>
      <p>Upload a photo of a plant leaf. A MobileNetV2 classifier trained on 38 disease
      classes returns a diagnosis with a confidence readout, plus a Grad-CAM map showing
      which part of the leaf it looked at.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
        diagnose = st.button("Diagnose", use_container_width=True)

with col2:
    with st.container(border=True):
        results_slot = st.empty()
        if uploaded is None:
            results_slot.markdown(
                "<div class='readout'>Upload a leaf photo to begin.</div>",
                unsafe_allow_html=True,
            )

if uploaded is not None and diagnose:
    pil_img = Image.open(uploaded).convert("RGB")
    with st.spinner("Analyzing leaf..."):
        html, overlay = predict_and_explain(pil_img)
    results_slot.markdown(html, unsafe_allow_html=True)

    if overlay is not None:
        with st.container(border=True):
            st.image(
                overlay,
                caption="Where the model looked (Grad-CAM)",
                use_container_width=True,
            )

with st.expander("How this works"):
    st.markdown(
        "1. **Upload** \u2014 you provide a leaf photo.\n"
        "2. **Preprocess** \u2014 resized to 224\u00d7224 and normalized for MobileNetV2.\n"
        "3. **Predict** \u2014 the model scores all 38 classes; the highest becomes the diagnosis.\n"
        "4. **Explain** \u2014 Grad-CAM highlights the leaf regions that most influenced that score."
    )

st.markdown(
    """
    <div class="disclaimer">
      This tool gives an AI-based preliminary read, not a substitute for expert agricultural
      diagnosis. Test accuracy on the PlantVillage dataset was 94.27% \u2014 real field photos
      may perform differently.
    </div>
    """,
    unsafe_allow_html=True,
)
