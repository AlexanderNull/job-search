from flask import abort, Flask, send_from_directory, request, jsonify
import os
from pymongo import MongoClient
import requests

from server.config import config
import server.db_tools
from server.job_provider import JobProvider
from server.train_model import ModelProvider


app = Flask(__name__, static_folder='../web-app/build')
client = MongoClient(config['mongo']['host'], config['mongo']['port'])
jobs_host = config['jobs_firebase_host']
historical_limit = config['historical_limit']
throttle_group_size = config['throttle_group_size']
throttle_duration = config['throttle_duration']
db_name = config['mongo']['data']['db_name']
table_name = config['mongo']['data']['table_name']
label_key = config['mongo']['data']['label_key']

db = client[db_name]
jobs_table = db[table_name]

job_provider = JobProvider(jobs_host, throttle_group_size, throttle_duration)
model_provider = ModelProvider()

# TODO: handle error condition of no saved embedding when run if you get around to it
@app.route('/api/model', methods=['POST'])
def trainModel():
    labeled_jobs = jobs_table.find({ '$and': [{ 'preferred': { '$exists': True } }, { 'text': { '$exists': True } }] })
    score = model_provider.train_model(labeled_jobs)
    return jsonify({ 'score': score })

@app.route('/api/model/predict', methods=['POST'])
def predict_text():
    text = request.get_json()
    if len(text) > 0:
        prediction = model_provider.predict(text)
        return jsonify({ label_key: prediction })

@app.route('/api/jobs/<int:job_id>', methods=['PUT'])
def updateJob(job_id):
    post_body = request.get_json()
    if label_key in post_body:
        update = jobs_table.update({ 'id': job_id }, { '$set': { label_key: post_body[label_key] }})
        if update['updatedExisting']:
            return jsonify(JobProvider.format_post(jobs_table.find_one({ 'id': job_id })))
        else:
            return abort(500)
    
    abort(400)

@app.route('/api/jobs/unlabeled')
def getJobs():
    unlabeled_jobs = list(jobs_table.find({ label_key: None }))
    if len(unlabeled_jobs) != 0:
        return jsonify([JobProvider.format_post(x) for x in unlabeled_jobs])
    else:
        oldest_id, newist_id = db_tools.get_jobs_range(jobs_table)
        new_jobs = job_provider.get_next_post(historical_limit, oldest_id, newest_id)
        if new_jobs is not None:
            jobs_table.insert_many(new_jobs)
            return jsonify([JobProvider.format_post(x) for x in new_jobs])
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
