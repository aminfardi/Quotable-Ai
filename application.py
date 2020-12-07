# application.py
import os.path, os
from flask import Flask, render_template, request, jsonify,url_for, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import QuoTable_Functions as qf
import re

# List of files for query
global file_list
global login_id
#add NBA files by default
file_list = ['NBA_Top_1000.csv', 'NBA_Top_200_Playoffs.csv']
# file_list = []
login_id = "testuser"
#file_list = ['NBA_Top_200.csv']
# file_list = []

application = Flask(__name__) # application 'app' is object of class 'Flask'
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///query.db'
db = SQLAlchemy(application)

#secret key
application.config['SECRET_KEY'] = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'

# Model
class QATable(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	question = db.Column(db.Text, nullable=False)
	answer = db.Column(db.Text)

	#def __repr__(self):
    #    return '<Query %r>' % self.id
	#querytime = db.column(db.DateTime, default=datetime.utcnow)

# import files
#from routes import *

@application.route('/', methods=['GET', 'POST'])
def home():
	# if 'username' in session:
	# 	username = session['username']
	# 	print ("Username is:", username)
	# if 'visits' in session:
	# 	session['visits'] = session.get('visits') + 1  # reading and updating session data
	# else:
	# 	session['visits'] = 1 # setting session data
	# 	print ("Visit is:", session['visits'])
	return render_template('index.html')


@application.route('/index', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

# @application.route('/answer_query')
# def answer_query():
# 	try:
# 		query_q = request.args.get('query_q', 0, type=str)
# 		if query_q == "":
# 			query_q = "What happens when I submit an empty query?"
# 		# print(query_q)
# 		# print ("I donnot know")
# 		# if query_q == "What player has the most career assists?":
# 		# return jsonify(result='John Stockton')
# 		# else:
# 		# return jsonify(result="No clue")
# 		# table = qf.AQuA_v1(query_q,df,"dataframe","dataframe","").to_csv(sep="|")
# 		# query_a = qf.predict(table, [query_q])
#
# 		#Moving to AQuA_v4
# 		[table2,query2]= qf.AQuA_v4(query_q,file_list,"csv_file","dataframe","")
# 		print (table2)
# 		table = table2.to_csv(sep="|")
# 		query_a = qf.predict(table, [query2])
# 		print("You asked:",query_q)
# 		print("Answer:",query_a)
# 		return jsonify(result=query_a)
# 	except Exception as e:
# 		return str(e)

@application.route('/answer_query')
def answer_query():
	try:
		query_q = request.args.get('query_q', 0, type=str)
			# print(query_q)
			# print ("I donnot know")
			# if query_q == "What player has the most career assists?":
			# return jsonify(result='John Stockton')
			# else:
			# return jsonify(result="No clue")
			# table = qf.AQuA_v1(query_q,df,"dataframe","dataframe","").to_csv(sep="|")
			# query_a = qf.predict(table, [query_q])
	#
			#check table size and decide whether to use AQuA
		[table2,query2]= qf.Table_Selector_v4(query_q,file_list,"csv_file","dataframe","")
		if len(table2.columns) * len(table2) < 200:
			print(table2)
			table = table2.to_csv(sep="|",index = False)
			print(table)
			print(query2)
			query_a = qf.predict(table, [query2])
		else:
			[table2,query2]= qf.AQuA_v5(query_q,file_list,"csv_file","dataframe","")
			print(table2)
			table = table2.to_csv(sep="|", index=False)
			print(table)
			print(query2)
			query_a = qf.predict(table, [query2])
		print("You asked:",query_q)
		print("Answer:",query_a)
		if query_a == "":
			return jsonify(result="I can not answer that.  Please try another query or upload more data tables.")
		else:
			return jsonify(result=query_a)
	except:
		return jsonify(result="I can not answer that.  Please try another query or upload more data tables.")

@application.route('/uploadLabel',methods=[ "GET",'POST'])
def uploadLabel():
    isthisFile=request.files.get('file')
	# path = request.files.getpath('file')
	# path = request.files['file'].file.name
    print("File is:", isthisFile)
    print("Filename is:", isthisFile.filename)


	#df = pd.read_csv(request.files.get('file'))
	#print df[0]
	# print("File path is:", path)
	# print("Filename 2 is:", isthisFile.file.name)
    # isthisFile.save("./"+isthisFile.filename)

@application.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        global df

        isthisFile=request.files.get('file')
        file_name = isthisFile.filename

        print ("File is", isthisFile)
        print ("File is", file_name)

        is_csv_file = re.compile("\.csv")
        is_xl_file = re.compile("\.xls")
        if is_csv_file.search(file_name):
            print (file_name, "is CSV file")
            df = pd.read_csv(request.files.get('file'))
        elif is_xl_file.search(file_name):
            print (file_name, "is XL file")
            df = pd.read_excel(request.files.get('file'))
        else:
            print ("Error! Wrong file type")
        print ("Loc0:", df.head(5))

        df.to_csv(isthisFile.filename)
        file_with_path = os.path.join(login_id,file_name)
        if os.path.isdir(login_id):
            print ("Directory exists")
        else:
            print ("Directory doesnot exist, creating it")
            os.mkdir(login_id)
        df.to_csv(file_with_path, index=False)

        # if file_name not in file_list:
        #     file_list.append(isthisFile.filename)
        #     print ("Files so far:", file_list)

        if file_with_path not in file_list:
            file_list.append(file_with_path)
            print ("Files so far:", file_list)

@application.route('/login', methods=['GET', 'POST'])
def login():
	global login_id
	try:
		login_id = request.args.get('login_id', 0, type=str)
		if login_id == "":
			login_id = "testuser"
	except Exception as e:
		return str(e)
	# if request.method == 'POST':
	# 	global login_id
	# 	default_id = "testuser"
	# 	login_id = request.form.get('login_id', default_id)
	print ("Login id:", login_id)
	# print ("Login id from form", request.form.get('login_id'))

if __name__ == '__main__':
    # '0.0.0.0' = 127.0.0.1 i.e. localhost
    # port = 5000 : we can modify it for localhost
    #application.run(host='0.0.0.0', port=5000, debug=True) # local webserver : application.run()
    application.run(debug=True) # local webserver : application.run()
