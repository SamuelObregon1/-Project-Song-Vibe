import flask
import os
import pickle
import pandas as pd
import re
from skimage import io
from skimage import transform

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import string

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


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')


df = pd.read_csv('https://raw.githubusercontent.com/SamuelObregon1/Project-Song-Vibe/main/DATA/music_data.csv')

app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = 'models/vectorizer.pkl'
path_to_text_classifier = 'models/text-classifier.pkl'
path_to_image_classifier = 'models/image-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)

with open(path_to_image_classifier, 'rb') as f:
    image_classifier = pickle.load(f)


def make_lower(a_string):
    return a_string.lower()

def remove_punctuation(a_string):    
    a_string = re.sub(r'[^\w\s]','',a_string)
    return a_string

def text_pipeline(input_string):
    input_string = make_lower(input_string)
    input_string = remove_punctuation(input_string)   
    return input_string

df['message_clean'] = df['lyrics']
df['message_clean'] = df['lyrics'].apply(text_pipeline)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
        #return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Get the input from the user.
        
        user_input_text = flask.request.form['user_input_text']
        #Created new list to store userinput
        usertext= []
        for i in user_input_text.split():
            usertext.append(i)

        Filter = '|'.join(usertext)
        data_frame = df[df['message_clean'].str.contains(Filter)]
        
        #data_frame = df[df['message_clean'].str.contains(usertext[0]) & df['message_clean'].str.contains(usertext[1]) & df['message_clean'].str.contains(usertext[2]) ]
        #data_frame = df[i for i in usertext if df['message_clean'].str.contains(i).bool()]
        #data_frame = df[df['message_clean'].str.contains(list1[j]) & df['message_clean'].str.contains(list1[j+1]) & df['message_clean'].str.contains(list1[j+2])]
        cd = data_frame['artist_name']
        vd = data_frame['genre']
        porky = " Placeholder"
        kilt = "PlaceHolder"
        if cd.empty == True:
            porky = "No Results Found"
        else:
            porky = cd.to_string(index=False)


        if vd.empty==True:
            kilt = " No Results Found"
        else:
            kilt = vd.to_string(index=False)
        #print(Filter)
       # print(data_frame)
        #print(data_frame['genre'] == 'pop', data_frame['genre'] == 'blues', data_frame['genre'] == 'rock', data_frame['genre'] == 'reggae', data_frame['genre'] == 'jazz', data_frame['genre'] == 'hip hop', data_frame['genre'] == 'country')

        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])
        # Make a prediction 
        predictions = model.predict(X)
        # Get the first and only value of the prediction.
        prediction = predictions[0]
        # Get the predicted probabs
        predicted_probas = model.predict_proba(X)
        predicted_proba = predicted_probas[0]
        # The first element in the predicted probabs is % genre
        genre = predicted_proba[0]

        return flask.render_template('index.html', 
            input_text=user_input_text,
            result=prediction,
            genre=genre,
            porky=porky,
            kilt = kilt)




@app.route('/input_values/', methods=['GET', 'POST'])
def input_values():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('input_values.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        var_one = flask.request.form['input_variable_one']
        var_two = flask.request.form['another-input-variable']
        var_three = flask.request.form['third-input-variable']

        list_of_inputs = [var_one, var_two, var_three]

        return(flask.render_template('input_values.html', 
            returned_var_one=var_one,
            returned_var_two=var_two,
            returned_var_three=var_three,
            returned_list=list_of_inputs))

    return(flask.render_template('input_values.html'))


@app.route('/images/')
def images():
    return flask.render_template('images.html')


@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('classify_image.html'))

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            # Read the image using skimage
            img = io.imread(file)

            # Resize the image to match the input the model will accept
            img = transform.resize(img, (28, 28))

            # Flatten the pixels from 28x28 to 784x0
            img = img.flatten()

            # Get prediction of image from classifier
            predictions = image_classifier.predict([img])

            # Get the value of the prediction
            prediction = predictions[0]

            return flask.render_template('classify_image.html', prediction=str(prediction))

    return(flask.render_template('classify_image.html'))


if __name__ == '__main__':
    app.run(debug=True)


