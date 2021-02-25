from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')


filename = 'model.pkl'
loaded_model    = pickle.load(open(filename, 'rb'))
lb_make  = pickle.load(open('label.pkl','rb'))
vectorizer  = pickle.load(open('vectorizer.pkl','rb'))


app = Flask(__name__)

all_stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def text_preprocessing(str_input):
  sentences = nltk.sent_tokenize(str_input)
  words = [lemmatizer.lemmatize(word) for word in sentences if word not in all_stopwords]
  words = [word.lower() for word in words if word.lower() not in  all_stopwords]
  string = ' '.join(words)
  string = re.sub(r'[^A-Za-z]',' ',string)
  return string

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

def make_prediction(strinput):
	  prediction = lb_make.inverse_transform(loaded_model.predict(vectorizer.transform([text_preprocessing(strinput)])))

	  predlist = prediction.tolist()

	  str1 = ''.join(predlist)

	  return str1

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/transform', methods=["POST"])
def transform():
	f = request.files['data_file']
	data = []
	with open(f) as file:
		csvfile = csv.reader(file)
		for row in csvfile:
			data.append(row)
	return render_template('data.html', data=data)
    """
    stream = f.read()

    #stream = io.StringIO(f.stream.read())
    #stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    #csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    #print(csv_input)

    stream.seek(0)
    result = stream.read()
    #result = transform(stream.read())

    df = pd.read_csv(StringIO(result), usecols=[1])
    
    
    df['prediction'] = df['Solution'].apply(make_prediction)
    #df['prediction'] = df.apply(make_prediction)

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
  
    return response
    """
    
if __name__ == "__main__":
    app.run(debug=True)
