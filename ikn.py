import os
import pickle
import streamlit as st
import pyLDAvis
import pyLDAvis.sklearn
import pandas as pd
from streamlit_option_menu import option_menu

# Set konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Topic Modelling IKN",
    layout="wide",
    page_icon="🌍"
)

# Menentukan direktori kerja
working_dir = os.path.dirname(os.path.abspath(__file__))

# Memuat model LDA yang telah disimpan
model_path = os.path.join(working_dir, 'ikn_model.sav')
with open(model_path, 'rb') as f:
    lda_model = pickle.load(f)

# Memuat dataset (opsional, jika ingin menampilkan sampel data)
data_path = os.path.join(working_dir, 'ikn_dataset.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = pd.DataFrame()

# Sidebar navigasi
with st.sidebar:
    selected = option_menu(
        'Analisis IKN',
        ['Visualisasi Topik', 'Dataset'],
        menu_icon='globe',
        icons=['bar-chart', 'database'],
        default_index=0
    )

# Halaman Visualisasi Topik
if selected == 'Visualisasi Topik':
    st.title("Visualisasi Topic Modelling IKN")
    
    st.write("Berikut adalah hasil analisis topic modelling menggunakan model LDA.")
    
    # Menampilkan visualisasi topik menggunakan pyLDAvis
    if 'vectorized_data' in locals():  # Pastikan ada data vektorisasi sebelumnya
        vis = pyLDAvis.sklearn.prepare(lda_model, vectorized_data, vectorizer)
        pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(pyLDAvis_html, width=1300, height=800)
    else:
        st.warning("Data vectorized tidak ditemukan. Pastikan preprocessing telah dilakukan sebelumnya.")

# Halaman Dataset
if selected == 'Dataset':
    st.title("Dataset Analisis IKN")
    
    if df.empty:
        st.warning("Dataset tidak ditemukan atau kosong.")
    else:
        st.write("Berikut adalah beberapa sampel data dari analisis topic modelling IKN:")
        st.dataframe(df.head(20))
