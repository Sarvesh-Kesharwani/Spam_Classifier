import streamlit as st
import pandas
import pickle

# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
sms = st.text_input('Enter the message')