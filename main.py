# import packages
# text preprocessing modules
from string import punctuation

import nltk
import numpy as np
import streamlit as st

# text preprocessing modules
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import joblib

import warnings

warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

# load stop words
stop_words = stopwords.words("english")


# function to clean the text
@st.cache
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers

    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text


# functon to make prediction
@st.cache
def make_prediction(review):
    # clearn the data
    # clean_review = text_cleaning(review)

    # load the model and make prediction
    # model = joblib.load("models/classifier_chain_model.pkl")

    # make prection
    # result = model.predict([clean_review])

    # check probabilities
    # probas = model.predict_proba([clean_review])
    # probability = "{:.2f}".format(float(probas[:, result]))
    result = "**javascript | html | css**"
    return result


# Set the app title
st.title("Auto Tagging Stack Overflow Questions")

model = st.radio(
     "Select the model to run : ",
     ('Label Powerset', 'Classifier Chains', 'Binary Relevance'))
if model == 'Label Powerset':
     st.write('Label Powerset is selected')
elif model == 'Classifier Chains':
    st.write('Classifier Chains model is selected')
elif model == 'Binary Relevance':
     st.write("Binary Relevance model is selected")
else:
    st.write("Please select the model")

# Declare a form to receive a movie's review
form = st.form(key="my_form")
review = form.text_input(label="Enter Stack overflow question for predicting tags :")
submit = form.form_submit_button(label="Make Prediction")

if submit:
    # make prediction from the input text
    result = make_prediction(review)

    # Display results of the NLP task
    st.header("Results")

    st.write(result)
