import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
import clip
from datetime import datetime

# ---- Paths ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "products_with_subcategory.csv")
NPZ_PATH = os.path.join(DATA_DIR, "clip_features.npz")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"

@st.cache_resource
def load_model():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    return model, preprocess

@st.cache_data
def load_index():
    df = pd.read_csv(CSV_PATH, sep=";")
    df["ID"] = df["ID"].astype(str)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    paths = npz["paths"]          # pvz /content/drive/.../4997.jpg
    feats = npz["features"].astype(np.float32)

    # Normalizavimas
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    # Iš path ištraukiamas ID pagal failo pavadinimą
    ids = []
    for p in paths:
        base = os.path.splitext(os.path.basename(str(p)))[0]
        ids.append(base)

    index_df = pd.DataFrame({"ID": ids, "path": paths})
    merged = index_df.merge(df, on="ID", how="inner")
    id_to_row = {ids[i]: i for i in range(len(ids))}
    emb = np.vstack([feats[id_to_row[i]] for i in merged["ID"].tolist()])

    return merged, emb

def first_image_url(urls: str) -> str:
    # CSV lauke "Image URLs" yra URL'ai atskirti '|'
    if not isinstance(urls, str) or not urls:
        return ""
    return urls.split("|")[0]

def encode_query_image(pil_img: Image.Image, model, preprocess) -> np.ndarray:
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)

# ---- UI ----
st.set_page_config(page_title="Batų paieška pagal nuotrauką", layout="wide")

# ---- CSS ----
st.markdown(
    """
    <style>
    /* Browse files */
    div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]):not([aria-label*="Clear"]) {
        background-color: #DECDBD !important;
        color: #382104 !important;
        border: 1px solid #382104 !important;
    }
    div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]):not([aria-label*="Clear"]):hover {
        background-color: #A6845E !important;
    }

    /* X mygtukas*/
    div[data-testid="stFileUploader"] button[aria-label*="Remove"],
    div[data-testid="stFileUploader"] button[aria-label*="Clear"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* INFO pranešimas */
    div.stAlert div[data-testid="stAlertContainer"][data-baseweb="notification"]{
        background-color: rgba(210,195,170,0.40) !important;
    }

    /* Tekstas INFO viduje */
    div.stAlert div[data-testid="stAlertContentInfo"] p{
        color: #3B2A1A !important;
    }

    /* Fonui */
    div.stAlert div[data-testid="stAlertContentInfo"],
    div.stAlert div[data-testid="stAlertContentInfo"] *{
        background: transparent !important;
        background-image: none !important;
    }

    /* Layout: mažesnis tarpas viršuje */
    .block-container { padding-top: 0rem; padding-bottom: 2rem; }

    /* Header permatomas */
    [data-testid="stHeader"] { background: transparent !important; }

    /* Fonas su tekstūra */
    [data-testid="stAppViewContainer"] {
      background-color: #EFE9E3 !important;
      background-image:
        radial-gradient(circle at 20% 15%, rgba(201,181,156,0.12), transparent 40%),
        radial-gradient(circle at 80% 30%, rgba(201,181,156,0.12), transparent 45%),
        radial-gradient(circle at 30% 85%, rgba(201,181,156,0.12), transparent 45%) !important;
      background-size: 24px 24px !important;
    }

    /* Kad pagrindinis turinio sluoksnis neturėtų savo fono */
    [data-testid="stMain"] {
      background: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {
      background: rgba(18, 24, 38, 0.95) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='margin-bottom:0;'>Batų paieška pagal nuotrauką</h1>", unsafe_allow_html=True)

# ---- Išdėstymas ----
left, right = st.columns([2, 1])

label_to_value = {
    "Moteriški ir vyriški": "Abu",
    "Moteriški": "Moterims",
    "Vyriški": "Vyrams",
}

with right:
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="uploader")
    top_k = st.slider("Kiek rezultatų rodyti", 1, 30, 10)
    gender_label = st.selectbox(
        "Kategorija",
        ["Moteriški ir vyriški", "Moteriški", "Vyriški"]
    )
    gender = label_to_value[gender_label]

with left:
    st.markdown("### Užklausa")
    if uploaded is None:
        st.info("Įkelkite bato nuotrauką ir pasirinkite, kiek panašių rezultatų rodyti.")
        st.stop()
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, width=220)

# ---- Modelio ir index užkrovimas ----
model, preprocess = load_model()
meta_df, emb = load_index()

# ---- Filtravimas pagal lyties pasirinkimą ----
if gender != "Abu":
    mask = meta_df["Category"].astype(str).eq(gender)
    meta_df_f = meta_df[mask].reset_index(drop=True)
    emb_f = emb[mask.to_numpy()]
else:
    meta_df_f = meta_df
    emb_f = emb
if len(meta_df_f) == 0:
    st.error("Pagal pasirinktą kategoriją nerasta įrašų.")
    st.stop()

# ---- Radimas ----
q = encode_query_image(query_img, model, preprocess)

nn = NearestNeighbors(n_neighbors=min(top_k, len(meta_df_f)), metric="cosine")
nn.fit(emb_f)
distances, indices = nn.kneighbors(q)

sims = 1.0 - distances[0]
idxs = indices[0]

st.subheader("Panašūs rezultatai")

# ---- Rezultatų išvedimas ----
cols = st.columns(5)
for i, (row_idx, sim) in enumerate(zip(idxs, sims)):
    row = meta_df_f.iloc[int(row_idx)]
    url = first_image_url(row.get("Image URLs", ""))
    title = f"{row.get('Name','')}"

    with cols[i % 5]:
        if url:
            st.image(url, caption=title, use_container_width=True)
        else:
            st.write(title)
            st.write("Nėra URL")

# ---- Rezultatų išsaugojimas į CSV ----
rows = []
for rank, (row_idx, sim) in enumerate(zip(idxs, sims), start=1):
    row = meta_df_f.iloc[int(row_idx)]
    rows.append({
        "rank": rank,
        "id": row.get("ID", ""),
        "category": row.get("Category", ""),
        "subcategory": row.get("Subcategory", ""),
        "price": row.get("Price", ""),
        "image_url": first_image_url(row.get("Image URLs", "")),
        "similarity": float(sim),
    })

out_df = pd.DataFrame(rows)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(RESULTS_DIR, "last_results.csv")
out_df.to_csv(out_path, index=False, encoding="utf-8")

st.success(f"Rezultatai išsaugoti: {out_path}")
