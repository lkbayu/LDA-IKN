import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Load model LDA dari file yang disimpan
with open("ikn_model.sav", "rb") as file:
    loaded_data = pickle.load(file)

# Pastikan semua komponen model LDA tersedia
lda_model = loaded_data.get("lda_model")
dictionary = loaded_data.get("dictionary")
bow_corpus = loaded_data.get("bow_corpus")

# Streamlit UI
st.sidebar.title("Menu Navigasi")
selected = st.sidebar.radio("Pilih Halaman:", ["Visualisasi Topik"])

# Halaman Visualisasi Topik
if selected == 'Visualisasi Topik':
    st.title("Visualisasi Topic Modelling IKN")
    st.write("Berikut adalah hasil analisis topic modelling menggunakan model LDA.")
    
    if lda_model and dictionary and bow_corpus:
        # Buat visualisasi dengan gensim
        vis = gensimvis.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)
        pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(pyLDAvis_html, width=1300, height=800)
    else:
        st.warning("Model atau data tidak ditemukan. Pastikan model telah dimuat dengan benar.")
