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

def remove_number(text):
    return re.sub(r"\d+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_whitespace_LT(text):
    return text.strip()

def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)

def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def stopwords_removal(text, stopwords_list):
    return " ".join([word for word in text.split() if word not in stopwords_list])

def normalize_text(text, normalization_dict):
    words = text.split()
    return " ".join([normalization_dict.get(word, word) for word in words])

def stemming_text(text, stemmer):
    return stemmer.stem(text)

def preprocess_text(df, text_column, stopwords_list, normalization_dict, stemmer):
    df[text_column] = df[text_column].astype(str).apply(remove_tweet_special)
    df[text_column] = df[text_column].apply(remove_number)
    df[text_column] = df[text_column].apply(remove_punctuation)
    df[text_column] = df[text_column].apply(remove_whitespace_LT)
    df[text_column] = df[text_column].apply(remove_whitespace_multiple)
    df[text_column] = df[text_column].apply(remove_single_char)
    df[text_column] = df[text_column].str.lower()
    df['cleaned_text'] = df[text_column].apply(lambda x: stopwords_removal(x, stopwords_list))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: normalize_text(x, normalization_dict))
    df['stemmed_text'] = df['cleaned_text'].apply(lambda x: stemming_text(x, stemmer))
    df['tokenized_text'] = df['stemmed_text'].apply(word_tokenize)
    return df

def main():
    st.title("Analisis Topic Modelling IKN dengan LDA")
    df = load_data()
    if df is not None:
        text_column = st.selectbox("Pilih kolom teks", df.columns)
        if st.button("Proses Data"):
            stopwords_list = set(stopwords.words('indonesian'))
            normalization_dict = {}
            stemmer = StemmerFactory().create_stemmer()
            df = preprocess_text(df, text_column, stopwords_list, normalization_dict, stemmer)
            st.write("Data setelah preprocessing:", df.head())
            dictionary = corpora.Dictionary(df['tokenized_text'])
            bow_corpus = [dictionary.doc2bow(text) for text in df['tokenized_text']]
            lda_model = LdaModel(bow_corpus, num_topics=5, id2word=dictionary, passes=10, alpha=0.5, random_state=37)
            topics = lda_model.print_topics(num_words=10)
            st.write("Topik yang ditemukan:")
            for topic in topics:
                st.write(topic)
            plt.figure(figsize=(15, 12))
            for i, topic in enumerate(topics):
                ax = plt.subplot(3, 3, i+1)
                words = topic[1]
                word_freq = {word.split('*')[1].strip().strip('"'): float(word.split('*')[0]) for word in words.split(' + ')}
                wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Topik {i+1}')
                ax.axis('off')
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(2)
            plt.tight_layout()
            st.pyplot(plt)
            lda_vis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)
            pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_vis_data)
            st.components.v1.html(pyLDAvis_html, width=1300, height=800)
if __name__ == "__main__":
    main()
