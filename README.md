![ALT text](assets/Web%20interface.png)



### Project Overview

This repository hosts a Python-based search engine and product recommendation system, developed to enhance the searchability and recommendation of products within Amazon's dataset. The application employs a variety of libraries to preprocess data and enable product searches through sophisticated text processing techniques and similarity measures. Additionally, it includes a user-friendly interface crafted with **Streamlit**, demonstrating the practical application of machine learning models.

#### Repository Structure

Contained within the repository are the following key files:
- `Amazon Product Predicting.py`: This is the primary Python script that contains all the functional components of the search engine and the Streamlit web interface.
- `amazon_product.csv`: This dataset file is essential as it contains the Amazon product details necessary for the search functionality.
- `assets`: This file contains some images used in this project
  
### Software and Library Requirements

This project is implemented in **Python** and necessitates the installation of several libraries, namely: Pandas, NumPy, NLTK, scikit-learn, Streamlit, and PIL (Pillow). 

### SSL Configuration

SSL is configured for unverified context use, an important measure to ensure secure connections in environments where SSL certificate verification might fail. This setup ensures the code operates smoothly without interruptions for all SSL connections during the session.

### Data Preprocessing

Upon loading the data from the [`amazon_product.csv`](./amazon_product.csv) file, the 'id' column was dropped, considering it was unnecessary for the analysis, thereby simplifying the dataset to only include relevant data.

### Text Processing Techniques

For text data handling, `SnowballStemmer` from NLTK was used to stem English words, simplifying them to their root form, which includes converting text to lowercase and splitting by whitespace to separate out individual words for stemming.

### Vectorisation and Similarity Calculation

Text data transformation into a matrix of TF-IDF features is performed using the `TfidfVectorizer`, specifically configured to filter out English stop words to optimise feature relevance. Following this, cosine similarity is calculated between the TF-IDF vectors of product descriptions to power the search functionality by quantifying textual similarity.

### Search Function Implementation

The core search functionality is encapsulated in a function that processes a user query by stemming and tokenising it, then computes cosine similarity between the query and each product. Products are then ranked based on their similarity scores, and the top 10 matches are returned to the user.

### Streamlit Web Application

The Streamlit framework is deployed to craft a straightforward and interactive web interface, which features:
- Displaying an image using functionalities provided by PIL’s `Image.open`.
- A text input field allowing users to enter a product name and a button to initiate the search.
- Display of the top 10 similar products based on the user’s query, utilising the search functionality described.

### Running the Application

To run the application, users should navigate to the project directory and initiate the script through Streamlit with the command `streamlit run Amazon Product Predicting.py` in the terminal. This will launch the web interface locally, enabling users to interact with the search engine.
Ending
