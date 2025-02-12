import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

# Download stopwords jika belum tersedia
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """Fungsi untuk membersihkan dan memproses teks."""
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    tokens = word_tokenize(text)  # Tokenisasi
    tokens = [word for word in tokens if word not in stop_words]  # Hapus stopwords
    return tokens

def main():
    st.title("Analisis Topic Modelling IKN dengan LDA")
    
    uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:", df.head())
        
        # Pastikan ada kolom teks
        text_column = st.selectbox("Pilih kolom teks untuk analisis", df.columns)
        
        if st.button("Proses Data"):
            st.write("Memproses data...")
            
            # Preprocessing teks
            df['processed_text'] = df[text_column].astype(str).apply(preprocess_text)
            
            # Membuat dictionary dan corpus BoW
            dictionary = corpora.Dictionary(df['processed_text'])
            corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
            
            # Menentukan jumlah topik
            num_topics = st.slider("Pilih jumlah topik", min_value=2, max_value=10, value=5)
            
            # Melatih model LDA
            lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
            
            # Tampilkan topik yang dihasilkan
            st.write("Topik yang dihasilkan:")
            for idx, topic in lda_model.show_topics(formatted=True, num_words=10):
                st.write(f"Topik {idx+1}: {topic}")
            
            # WordCloud untuk setiap topik
            st.write("### WordCloud untuk Setiap Topik")
            cols = st.columns(num_topics)
            for i in range(num_topics):
                words = dict(lda_model.show_topic(i, 20))
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(words)
                
                with cols[i % len(cols)]:
                    st.image(wordcloud.to_array(), caption=f"Topik {i+1}")
            
            # Visualisasi dengan pyLDAvis
            st.write("### Visualisasi Interaktif dengan pyLDAvis")
            vis = gensimvis.prepare(lda_model, corpus, dictionary)
            html_string = pyLDAvis.prepared_data_to_html(vis)
            st.components.v1.html(html_string, width=1000, height=800)

if __name__ == "__main__":
    main()
