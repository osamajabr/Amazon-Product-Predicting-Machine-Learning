import pandas as pd
import numpy as np
import ssl
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Set up SSL certificates for secure downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load the dataset
data = pd.read_csv('/Users/osamajabr/Desktop/Amazon Product/amazon_product.csv')

# Remove unnecessary columns
data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    # Manually split the text into words based on whitespace
    tokens = text.lower().split()
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
def cosine_sim(text1, text2):
    # Concatenate text from the tokens for vectorization
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    # Vectorize the text and calculate cosine similarity
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function using the cosine similarity
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
    return results

# Web app setup using Streamlit
img = Image.open('/Users/osamajabr/Desktop/Amazon Product/img.png')
st.image(img, width=600)
st.title("Search Engine and Product Recommendation System on Amazon Data")
query = st.text_input("Enter Product Name")
submit = st.button('Search')
if submit:
    res = search_products(query)
    st.write(res)
