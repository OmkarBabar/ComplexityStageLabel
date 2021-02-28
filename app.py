from flask import Flask, make_response, request, render_template,jsonify
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
import codecs
import os
import flask_restful as restful
from werkzeug.utils import secure_filename
import boto3

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

filename = 'model.pkl'
loaded_model    = pickle.load(open(filename, 'rb'))
lb_make  = pickle.load(open('label.pkl','rb'))
vectorizer  = pickle.load(open('vectorizer.pkl','rb'))

S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

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

def transform(text_file_contents):
	return text_file_contents.replace("=", ",")

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/data', methods=['GET','POST'])
def data():
	if request.method == 'POST':
		
		"""
		print(ROOT_PATH)
		
		print('S3_BUCKET',S3_BUCKET)
		print('AWS_ACCESS_KEY_ID',AWS_ACCESS_KEY_ID)
		print('AWS_SECRET_ACCESS_KEY',AWS_SECRET_ACCESS_KEY)
		
		#s3 = boto3.client('s3')
		s3 = boto3.resource('s3')
		
		#client = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
		
		if s3:
			print('Success..............')
		else:
			print('Failed..............')
	
		file = request.files['csvfile']
		
		filename = secure_filename(file.filename)
		
		#s3.Bucket(S3_BUCKET).put_object(Key=filename,Body=file)
		
		s3.Bucket(S3_BUCKET).Object(file.filename).put(Body=file.read())
		
		return '<h1>success</h>'
		"""
		
		print("ggggggggg", request.files)
		file = request.files['csvfile']
		file.save(os.path.join(ROOT_PATH, file.filename))
		
		stream = io.TextIOWrapper(file.stream._file, "UTF8", newline=None)

		stream.seek(0)
		result = transform(stream.read())
		
		df = pd.read_csv(StringIO(result))
		
		df['Prediction'] = df['Solution'].apply(make_prediction)
		
		response = make_response(df.to_csv())
		#response = make_response(result)
		response.headers["Content-Disposition"] = "attachment; filename=result.csv"
		return response
		

if __name__ == "__main__":
	app.run(host='0.0.0.0', port = 8080)
