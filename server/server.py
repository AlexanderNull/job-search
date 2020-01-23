import os
from flask import Flask, send_from_directory, request
from pymongo import MongoClient

app = Flask(__name__, static_folder='../web-app/build')

@app.route('/api/jobs', methods=['POST'])
def loadJobs():
    pass

@app.route('/api/jobs')
def getJobs():
    # TODO: get latest unlabeled jobs and return json
    pass

@app.route('/api/jobs/<id>', methods=['PUT'])
def labelJob(id):
    # TODO: use provided id to determine which record to update, apply match criteria to record
    pass

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def reactApp(path):
    if path != '' and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)