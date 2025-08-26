import streamlit as st
import torch
import timm
from PIL import Image
import numpy as np
import plotly.express as px
from torchvision import transforms
from src.utils.labels import CIFAR10_LABELS

st.set_page_config(page_title="ViT CIFAR-10 Demo", page_icon="🧠", layout="wide")
st.title("🧠 Vision Transformer – CIFAR-10 Demo")

st.write("""
Upload an image, and the ViT model will predict a CIFAR-10 class.  
Place your trained weights at `results/best_vit.pt` (from `src/train.py`).  
If no weights are found, the app will use **random weights** (predictions will be nonsense).
""")

@st.cache_resource
def load_model(model_name="vit_tiny_patch16_224", num_classes=10, ckpt_path="results/best_vit.pt"):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            st.success(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            st.warning(f"Could not load checkpoint: {e}")
    model.eval()
    return model

import os
model = load_model()

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
])

uploaded = st.file_uploader("Upload an image (will be resized to 224x224)", type=["png","jpg","jpeg"])

def predict(img: Image.Image):
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs

col1, col2 = st.columns([1,1])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1.image(img, caption="Input", use_column_width=True)
    probs = predict(img)
    topk = probs.argsort()[-5:][::-1]
    top_labels = [CIFAR10_LABELS[i] for i in topk]
    fig = px.bar(x=top_labels, y=probs[topk], labels={"x":"Class","y":"Probability"}, title="Top-5 Predictions")
    fig.update_layout(yaxis_range=[0,1])
    col2.plotly_chart(fig, use_container_width=True)

    st.caption("Note: Attention overlay is a placeholder. Replace with your rollout/grad-cam for ViT.")
else:
    st.info("Upload an image to see predictions.")

st.markdown("---")
st.subheader("How to run locally")
st.code("""
pip install -r requirements.txt
streamlit run streamlit_app.py
""")
