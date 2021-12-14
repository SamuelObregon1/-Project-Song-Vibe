import flask
import os
import pickle
from flask.templating import render_template
import pandas as pd
import re
from skimage import io
from skimage import transform
import seaborn as sns
import nltk
import string
import json
import pandas as pd

from dash import Dash
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from flask import Flask


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')


df = pd.read_csv('https://raw.githubusercontent.com/SamuelObregon1/Project-Song-Vibe/main/DATA/music_data.csv')

app = flask.Flask(__name__, template_folder='templates')

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
df['mood_feeling']=df.iloc[0:28372,7:29].idxmax(axis=1)
df['max value'] = df.iloc[0:28372,7:29].max(axis=1)


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
        list1 = re.split('\s+', user_input_text)    
        print(list1)
        j=0
        
        #data_frame = df[df['message_clean'].str.contains(Filter)]
        
        #data_frame = df[x for x in usertext if df['message_clean'].str.contains(x).any()]
        #data_frame = df[df['message_clean'].str.contains(list1[j]) & df['message_clean'].str.contains(list1[j+1]) & df['message_clean'].str.contains(list1[j+2])]
        data_frame = df[df['message_clean'].str.contains(list1[j]) & df['message_clean'].str.contains(list1[j+1]) & df['message_clean'].str.contains(list1[j+2])]
        cd = data_frame['artist_name']
        vd = data_frame['genre']
        gg = data_frame['max value']
        db = data_frame['mood_feeling']
        ll = data_frame['track_name']

        print(gg)

        artists =[]
        genre = []
        song = []
        final = []
        gate = []
        rest = []
        boat = []
        
        list1 = re.split('\s+', user_input_text)    
        print(list1)
        j=0
        for x in gg:
            gate.append(x)
        
        for x in db:
            rest.append(x)

        for x in cd:
            artists.append(x)

        for x in vd:
            genre.append(x)
        
        for x in ll:
            song.append(x)

        i = 0
        for x in artists:
            final.append("Artist: " + artists[i] + ", Genre: " + genre[i] + ", Song: " + song[i])
            i+=1
    
        i = 0
        for x in gate:
            boat.append("Artist: " + artists[i]+ ", " + "Mood: " + rest[i])
            i+=1
        print(rest)
        #print(data_frame)
        #print(data_frame['genre'] == 'pop', data_frame['genre'] == 'blues', data_frame['genre'] == 'rock', data_frame['genre'] == 'reggae', data_frame['genre'] == 'jazz', data_frame['genre'] == 'hip hop', data_frame['genre'] == 'country')

        #fig = px.pie(data_frame, values='max value', names='mood_feeling')
        #fig.show()

        print(final)
        data_frame.head()
    
        return flask.render_template('index.html', 
            input_text=user_input_text,
            final = final,
            gate = gate,
            the_vibes = json.dumps(gate),
            the_labels = json.dumps(boat)
            )

@app.route("/chart/")
def chart():
    return render_template('chart.html')

@app.route('/images/')
def images():
    return flask.render_template('images.html')


@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')



if __name__ == '__main__':
    app.run(debug=True)