import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.sklearn
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Fungsi preprocessing teks
def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter selain huruf dan spasi
    return text

# Fungsi untuk menampilkan topik LDA
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

# Streamlit UI
st.title("Analisis Topic Modelling IKN dengan LDA")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Awal:")
    st.write(df.head())
    
    # Pilih kolom teks
    text_column = st.selectbox("Pilih kolom teks:", df.columns)
    df['clean_text'] = df[text_column].astype(str).apply(clean_text)
    
    # Tampilkan contoh hasil preprocessing
    st.write("### Data Setelah Preprocessing:")
    st.write(df[['clean_text']].head())
    
    # Buat CountVectorizer
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    dtm = vectorizer.fit_transform(df['clean_text'])
    
    # Pilih jumlah topik
    num_topics = st.slider("Pilih jumlah topik:", 2, 10, 5)
    
    # Train LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)
    
    # Tampilkan hasil topik
    st.write("### Topik yang Ditemukan:")
    topics = display_topics(lda_model, vectorizer.get_feature_names_out(), 10)
    for i, topic in enumerate(topics):
        st.write(f"**Topik {i+1}:** {topic}")
    
    # Visualisasi WordCloud tiap topik
    st.write("### WordCloud Tiap Topik:")
    fig, axes = plt.subplots(1, num_topics, figsize=(20, 5))
    for i, topic in enumerate(lda_model.components_):
        wc = WordCloud(background_color='white').generate(" ".join([vectorizer.get_feature_names_out()[j] for j in topic.argsort()[:-20 - 1:-1]]))
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis("off")
        axes[i].set_title(f"Topik {i+1}")
    st.pyplot(fig)
    
    # Visualisasi dengan pyLDAvis
    st.write("### Visualisasi Interaktif LDA:")
    lda_vis = pyLDAvis.sklearn.prepare(lda_model, dtm, vectorizer)
    with st.expander("Lihat Visualisasi LDA"):
        st.components.v1.html(pyLDAvis.prepared_data_to_html(lda_vis), height=800)