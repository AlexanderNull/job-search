import os
from flask import abort, Flask, send_from_directory, request, jsonify
from pymongo import MongoClient
import requests

from job_provider import JobProvider

app = Flask(__name__, static_folder='../web-app/build')
client = MongoClient('localhost', 27107)
db = client['job-search-database']

jobs_host = 'https://hacker-news.firebaseio.com/v0'
historical_limit = 2

job_provider = JobProvider(jobs_host, historical_limit)

@app.route('/api/jobs', methods=['POST'])
def loadJobs():
    # jobsResponse = requests.get(f'{jobs_host}/v4beta1/jobs:search')
    pass

@app.route('/api/jobs')
def getJobs():
    new_jobs = job_provider.get_next_historical()
    if new_jobs is not None:
        return jsonify(new_jobs)
    else:
        abort(404)

@app.route('/api/jobs/<id>', methods=['PUT'])
def labelJob(id):
    # TODO: use provided id to determine which record to update, apply match criteria to record
    pass

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def reactApp(path):
    if path != '' and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)