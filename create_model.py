import pandas as pd
import pickle

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import string
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')

def make_lower(a_string):
    return a_string.lower()

def remove_punctuation(a_string):    
    a_string = re.sub(r'[^\w\s]','',a_string)
    return a_string

def text_pipeline(input_string):
    input_string = make_lower(input_string)
    input_string = remove_punctuation(input_string)   
    return input_string


df = pd.read_csv('https://raw.githubusercontent.com/SamuelObregon1/Project-Song-Vibe/main/DATA/music_data.csv')

df['message_clean'] = df['lyrics']

df['message_clean'] = df['lyrics'].apply(text_pipeline)

X = df['message_clean'].values

y = df['genre'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Initialize our vectorizer
vectorizer = TfidfVectorizer()

# This makes your vocab matrix
vectorizer.fit(X)

# This transforms your documents into vectors.
X = vectorizer.transform(X)

# Initalize our model.
model = MultinomialNB(alpha=.025)

# Fit our model with our training data.
model.fit(X_train, y_train)

# Make new predictions of our testing data. 
y_pred = model.predict(X_test)

# Make predicted probabilites of our testing data
y_pred_proba = model.predict_proba(X_test)
    

# Save our vectorizer and model.
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb') )
pickle.dump(model, open('models/text-classifier.pkl', 'wb') )


