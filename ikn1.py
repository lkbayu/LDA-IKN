import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import gensim
from gensim import corpora
from gensim.models import LdaModel
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def remove_tweet_special(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\\w+:\/\/\\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

def preprocess_text(df, text_column, stopwords_list, normalization_dict, stemmer):
    if st.button("Case Cleaning"):
        df[text_column] = df[text_column].astype(str).apply(remove_tweet_special)
        df[text_column] = df[text_column].apply(lambda x: re.sub(r"\d+", "", x))
        df[text_column] = df[text_column].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
        df[text_column] = df[text_column].apply(lambda x: re.sub(r"\s+", " ", x).strip())
        df[text_column] = df[text_column].apply(lambda x: re.sub(r"\b[a-zA-Z]\b", "", x))
        st.write("Case Cleaning Done!", df.head())
    
    if st.button("Case Folding"):
        df[text_column] = df[text_column].str.lower()
        st.write("Case Folding Done!", df.head())
    
    if st.button("Stopword Removal"):
        df[text_column] = df[text_column].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords_list]))
        st.write("Stopword Removal Done!", df.head())
    
    if st.button("Normalization"):
        df[text_column] = df[text_column].apply(lambda x: " ".join([normalization_dict.get(word, word) for word in x.split()]))
        st.write("Normalization Done!", df.head())
    
    if st.button("Stemming"):
        df[text_column] = df[text_column].apply(lambda x: stemmer.stem(x))
        st.write("Stemming Done!", df.head())
    
    if st.button("Tokenization"):
        df['tokenized_text'] = df[text_column].apply(word_tokenize)
        st.write("Tokenization Done!", df[['tokenized_text']].head())
    
    if st.button("Bag of Words (BoW)"):
        dictionary = corpora.Dictionary(df['tokenized_text'])
        bow_corpus = [dictionary.doc2bow(text) for text in df['tokenized_text']]
        st.write("Bag of Words Done!", dictionary.token2id)
    
    return df

def main():
    st.title("Analisis Topic Modelling IKN dengan LDA")
    df = load_data()
    if df is not None:
        text_column = st.selectbox("Pilih kolom teks", df.columns)
        stopwords_list = set(stopwords.words('indonesian'))
        normalization_dict = {}
        stemmer = StemmerFactory().create_stemmer()
        df = preprocess_text(df, text_column, stopwords_list, normalization_dict, stemmer)
        
        if st.button("Modelling LDA"):
            dictionary = corpora.Dictionary(df['tokenized_text'])
            bow_corpus = [dictionary.doc2bow(text) for text in df['tokenized_text']]
            lda_model = LdaModel(bow_corpus, num_topics=5, id2word=dictionary, passes=10, alpha=0.5, random_state=37)
            topics = lda_model.print_topics(num_words=10)
            st.write("Topik yang ditemukan:")
            for topic in topics:
                st.write(topic)
        
        if st.button("Wordcloud"):
            plt.figure(figsize=(15, 12))
            for i, topic in enumerate(lda_model.print_topics(num_words=10)):
                ax = plt.subplot(3, 3, i+1)
                words = topic[1]
                word_freq = {word.split('*')[1].strip().strip('"'): float(word.split('*')[0]) for word in words.split(' + ')}
                wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Topik {i+1}')
                ax.axis('off')
            plt.tight_layout()
            st.pyplot(plt)
        
        if st.button("pyLDAvis"):
            lda_vis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)
            pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_vis_data)
            st.components.v1.html(pyLDAvis_html, width=1300, height=800)

if __name__ == "__main__":
    main()
