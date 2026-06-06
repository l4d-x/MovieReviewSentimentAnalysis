# 🎬 Movie Review Sentiment Classifier

A deep learning NLP pipeline that classifies 50,000 IMDB movie reviews as Positive or Negative, achieving **85.59% test accuracy** and **0.86 F1-score**.

## 🚀 Live Demo

**[Try it on Streamlit →](https://moviereviewsentimentanalysis-l4dx.streamlit.app/)**

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 85.59% |
| **F1 Score** | 0.86 |
| Parameters | 1.41M |
| Vocabulary | 10,000 words |

## 🏗️ Architecture

```
Raw Review Text
    ↓
Preprocessing (HTML strip, lowercase, Keras padding)
    ↓
Embedding Layer (10,000 vocab × 128-dim)
    ↓
LSTM Layer (128 units)
    ↓
SpatialDropout1D (overfitting prevention)
    ↓
Dense → Sigmoid → Positive/Negative
```

## 🔧 Technical Details

| Component | Technology |
|-----------|-----------|
| **Framework** | TensorFlow / Keras |
| **Architecture** | LSTM (1.41M parameters) |
| **Embeddings** | Trainable, 128-dim |
| **Vocabulary** | 10,000 words |
| **Max Sequence Length** | 200 tokens |
| **Dataset** | IMDB 50K reviews (25K train / 25K test) |
| **Preprocessing** | NLTK, HTML stripping, lowercasing |
| **Deployment** | Streamlit |

## 💡 Key Implementation Details

- **Text Preprocessing**: HTML tag stripping, lowercasing, and Keras sequence padding (max length 200) to reduce input noise
- **Regularization**: SpatialDropout1D applied to embedding layer to prevent overfitting on a 50K dataset
- **Model Serialization**: Saved Keras model and tokenizer for reusable inference
- **Deployment**: Interactive Streamlit web app for real-time predictions

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/l4d-x/MovieReviewSentimentAnalysis.git
cd movie-sentiment-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📝 Usage

```python
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Predict
from tensorflow.keras.preprocessing.sequence import pad_sequences
text = ["This movie was absolutely fantastic!"]
seq = tokenizer.texts_to_sequences(text)
padded = pad_sequences(seq, maxlen=200)
prediction = model.predict(padded)
print("Positive" if prediction[0][0] > 0.5 else "Negative")
```

## 🧠 What I Learned

- LSTM networks effectively capture sequential dependencies in text
- SpatialDropout1D is more effective than regular Dropout for NLP embeddings
- Vocabulary size of 10K provides a good balance between coverage and model size
- Proper text preprocessing (HTML stripping, lowercasing) significantly improves generalization
