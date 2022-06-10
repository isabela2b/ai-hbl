from flask import Flask, jsonify, request
import shutil, requests, json

from extract import extract_compare
import os
from datetime import datetime

"""import logging
from logging.handlers import RotatingFileHandler"""

# Allowed extension you can set your own

app = Flask(__name__)

#app routes will have to be for user interface of table extraction and learning
@app.route("/")
def home():
    return "Server Live"

@app.route("/compare", methods = ['POST'])
def compare():
    """
    Receives files, extracts data and pushes it to the API endpoint for comparison.
    """
    predictions = {}
    params = request.args
    if (params == None):
        params = request.args

    # if parameters are found, return a prediction
    if (params != None):
        #try:
        user_id = request.form['user_id']
        process_id = request.form['process_id']

        if "file" in request.form:
            file_url = request.form['file']
            response = requests.get(file_url)
            filename = file_url.rsplit('/', 1)[1]
            return extract_compare(response.content, filename, user_id, process_id)
        else:
            file_obj = request.files.getlist('file')#[0]
            if file_obj[0].filename != '':
                return extract_compare(file_obj, user_id, process_id)
            else:
                return "No file submitted."

        """except Exception as ex:
                                    return str(ex)"""

if __name__ == '__main__':
    #alogging.basicConfig(handlers=[RotatingFileHandler('logs/error.log', maxBytes=100000, backupCount=10)], level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    app.run(debug=True)