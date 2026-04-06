import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Config ──────────────────────────────────────────────────────────────────
MAX_LEN = 200

# ── Load model & tokenizer (cached so they load only once) ──────────────────
@st.cache_resource
def load_artifacts():
    model = load_model("sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# ── Text cleaning (same logic as the notebook) ───────────────────────────────
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)   # keep only letters
    text = text.lower().strip()
    return text

# ── Prediction ───────────────────────────────────────────────────────────────
def predict_sentiment(review, model, tokenizer):
    cleaned  = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_LEN, truncating="post")
    prob     = model.predict(padded, verbose=0)[0][0]
    label    = "POSITIVE 😊" if prob > 0.5 else "NEGATIVE 😞"
    return label, float(prob)

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type any movie review below and the LSTM model will classify it as **Positive** or **Negative**.")

model, tokenizer = load_artifacts()

review_input = st.text_area("Enter a movie review:", height=150,
                             placeholder="e.g. This movie was absolutely fantastic!")

if st.button("Analyze Sentiment", type="primary"):
    if not review_input.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_sentiment(review_input, model, tokenizer)

        # Result card
        color = "#2ecc71" if "POSITIVE" in label else "#e74c3c"
        st.markdown(
            f"""
            <div style="padding:1.5rem; border-radius:10px; background:{color}22;
                        border-left: 5px solid {color}; margin-top:1rem;">
                <h3 style="color:{color}; margin:0">{label}</h3>
                <p style="margin:0.5rem 0 0 0; color:#555">
                    Confidence: <strong>{confidence:.2%}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar
        st.progress(confidence if "POSITIVE" in label else 1 - confidence)

# ── Example reviews ──────────────────────────────────────────────────────────
with st.expander("Try example reviews"):
    examples = [
        "This movie was absolutely fantastic! The storyline kept me hooked till the very end.",
        "Worst movie I have ever watched. Complete waste of time and money.",
        "The acting was brilliant and the direction was phenomenal. A must watch!",
        "Boring storyline, bad acting and predictable ending. Very disappointing.",
    ]
    for ex in examples:
        if st.button(ex[:70] + "…", key=ex):
            st.session_state["example"] = ex
            st.rerun()

# Pre-fill text area from example click
if "example" in st.session_state:
    st.text_area("Selected example:", value=st.session_state["example"], key="prefill")
