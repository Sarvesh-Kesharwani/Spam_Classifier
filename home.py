import streamlit as st
import pandas
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

st.title('Email/SMS Spam Classifier')
input_sms = st.text_input('Enter the message')

def transform_text(text):
    translator = str.maketrans('', '', string.punctuation)
    ps = PorterStemmer()

    text = text.lower()
    text = nltk.word_tokenize(text)
    clean_text=[]
    for word in text:
        if word not in stopwords.words('english'):
            word = word.translate(translator)
            if word.isalnum():
                clean_text.append(ps.stem(word))
    return ' '.join(clean_text)

tfidf = pickle.load(open(r'C:\Users\sarvesh.kesharwani\Pictures\DSC\Spam_Classifier\artifacts\tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open(r'C:\Users\sarvesh.kesharwani\Pictures\DSC\Spam_Classifier\artifacts\model_mnb.pkl', 'rb'))

if st.button('Predict'):
    input_sms = transform_text(input_sms)
    vectored_sms = tfidf.transform([input_sms])
    result = model.predict(vectored_sms)[0]
    if result == 1:
        st.header("Spam!")
    else:
        st.header("Not Spam")

tfidf = pickle.load(open(r'C:\Users\sarvesh.kesharwani\Pictures\DSC\Spam_Classifier\artifacts\tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open(r'C:\Users\sarvesh.kesharwani\Pictures\DSC\Spam_Classifier\artifacts\model_mnb.pkl', 'rb'))


