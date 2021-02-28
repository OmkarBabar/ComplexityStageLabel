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

filename = 'model.pkl'
loaded_model    = pickle.load(open(filename, 'rb'))
lb_make  = pickle.load(open('label.pkl','rb'))
vectorizer  = pickle.load(open('vectorizer.pkl','rb'))

S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = r'D:\upload'

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
		
		print(ROOT_PATH)
		
		print('S3_BUCKET',S3_BUCKET)
		print('AWS_ACCESS_KEY_ID',AWS_ACCESS_KEY_ID)
		print('AWS_SECRET_ACCESS_KEY',AWS_SECRET_ACCESS_KEY)
		
		s3 = boto3.client('s3')
		
		#client = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
		
		if s3:
			print('Success..............')
		else:
			print('Failed..............')
	
		file = request.files['csvfile']
		
		filename = secure_filename(file.filename)
		
		s3.Bucket(S3_BUCKET).put_object(Key=filename,Body=file)
		
		#presigned_post = s3.generate_presigned_post(Bucket = S3_BUCKET,Key = filename)
		
		if presigned_post:
			print("Load Sucee...............")
		else:
			print("Load Fail...............")
		#s3.upload_fileobj(result,S3_BUCKET,filename)
		
		return '<h1>success</h>'
		
		"""
		session = boto3.Session( aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
		s3 = session.resource('s3')
		
		s3.Bucket(S3_BUCKET).put_object(Key=filename,Body=request.files['csvfile'])
		"""
		
		
		
		"""
		#print("ggggggggg", request.files)
		#file = request.files['csvfile']
		#file.save(os.path.join(UPLOAD_FOLDER, file.filename))
		
		f = request.files['csvfile']
		print("filename ", f.filename)
		if not f:
			return "No file"
		
		stream = io.TextIOWrapper(f.stream._file, "UTF8", newline=None)
		
		csv_input = csv.reader(stream)
		
		print(csv_input)
		for row in csv_input:
			print(row)
			
		stream.seek(0)
		result = transform(stream.read())

		response = make_response(result)
		response.headers["Content-Disposition"] = "attachment; filename=omiresult.csv"
		return response
		"""
	
		"""
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		df = pd.read_csv(os.path.join(UPLOAD_FOLDER, sorted(os.listdir(app.config['UPLOAD_FOLDER']))[0]))
		
		df['Prediction'] = df['Solution'].apply(make_prediction)
		
		response = make_response(df.to_csv())
		response.headers["Content-Disposition"] = "attachment; filename=result.csv"
		return response
		"""
		
		"""
		csvfile = csv.reader(file)
		
		filename = csvfile.filename
		
		os.mkdir(UPLOAD_FOLDER)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		df = pd.read_csv(os.path.join(UPLOAD_FOLDER, sorted(os.listdir(app.config['UPLOAD_FOLDER']))[0]))
		df['Prediction'] = df['Solution'].apply(make_prediction)
		
		response = make_response(df.to_csv())
		response.headers["Content-Disposition"] = "attachment; filename=result.csv"
		return response
		"""
		"""
		stream = io.TextIOWrapper(file.stream._file, "UTF8", newline=None)
		#stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
		csv_input = csv.reader(stream)
		print(csv_input)
		data =[]
		for row in csv_input:
			print(row)
			data.append(row)
		
		#stream.seek(0)
		result = transform(stream.read())
		
		df = pd.read_csv(StringIO(result))
		#df = pd.read_csv(result)
		
		df['Prediction'] = df.apply(make_prediction)
		
		response = make_response(df.to_csv())
		response.headers["Content-Disposition"] = "attachment; filename=result.csv"
		return response
		"""

if __name__ == "__main__":
	app.debug = True
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port = port)
	#app.run(debug=True)
