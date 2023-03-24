import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the Doc2Vec model
d2v_model = Doc2Vec.load('models/doc2vec_model_v5000_w10_e50.bin')
d2v_model.vector_size = 5000
# Load the Logistic Regression model
lr_model = joblib.load('models/model_lr.pkl')


# Define a function to preprocess the message
def preprocess_message(text):
    # Remove unnecessary characters
    if text is None or pd.isnull(text):
        return ''

    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    return tokens


# Define a function to infer the Doc2Vec embeddings for a message
def infer_embeddings(text):
    tokens = preprocess_message(text)
    tokens_str = ' '.join(tokens)
    embedding = d2v_model.infer_vector(preprocess_message(tokens_str), epochs=50)

    return embedding


# Define a function to predict if a message is abusive or not
def predict_abusive(text, threshold=0.5):
    # Infer the Doc2Vec embeddings for the message
    text = str(text)  # Infer the Doc2Vec embeddings for the message
    embedding = infer_embeddings(text)

    # Reshape the embedding to have 5000 features
    embeddings = embedding.reshape(1, -1)
    # Make predictions using the Logistic Regression model
    prediction = lr_model.predict(embeddings)
    # prediction = lr_model.predict(embeddings)
    # probabilities = lr_model.predict_proba(embedding)
    # prediction = 1 if probabilities[0][1] > threshold else 0

    # Get the probability of the prediction being 1 (abusive)
    proba = lr_model.predict_proba(embeddings)[:, 1]

    return prediction[0], proba[0]


# Define the Streamlit app
def main():
    # Set the page title and favicon
    st.set_page_config(page_title='SUPERCELL Moderator', page_icon=':guardsman:')

    supercell_logo = 'empty_crown.png'
    st.image(supercell_logo, width=200)

    # Define the app header
    st.title('Enhancing Game Chat Moderation with AI')

    # st.markdown(
    #     'Please add a big chat sentence as the Doc2Vec model was trained using 5000 vector_size')

    # Create a text box for user input
    user_input = st.text_input('Start the chat...', '')

    # Check if the user has entered any input
    if user_input:
        # Predict if the message is abusive or not
        prediction, proba = predict_abusive(user_input)

        # Display the prediction result
        if prediction == 1:
            st.error('This chat should be moderated!')
        else:
            st.success('You can skip the chat!')

        st.text(f"Risk Score: {proba:.2f}")

    st.sidebar.title("Team Info")
    st.sidebar.write("Team Name: Passauer Mavericks")
    st.sidebar.write("Team Members:")
    st.sidebar.write("- Hrishikesh Jadhav")
    st.sidebar.write("- Yash Dalal")
    st.sidebar.write("STARTHACK")
    st.sidebar.markdown("---")
    st.sidebar.write("Task by: SUPERCELL")
    # Display the risk score


if __name__ == '__main__':
    main()
