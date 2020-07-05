from flask import abort, Flask, send_from_directory, request, jsonify
import os
from pymongo import MongoClient
import requests

from server.config import config
from server.db_tools import get_jobs_range
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
settings_table_name = config['mongo']['settings']['table_name']
sequence_key = "sequence_length"

db = client[db_name]
jobs_table = db[table_name]
settings_table = db[settings_table_name]

job_provider = JobProvider(jobs_host, throttle_group_size, throttle_duration)
model_provider = ModelProvider()

# TODO: handle error condition of no saved embedding when run if you get around to it
@app.route('/api/model', methods=['POST'])
def trainModel():
    params = request.get_json()
    labeled_jobs = jobs_table.find({ '$and': [{ 'preferred': { '$exists': True } }, { 'text': { '$exists': True } }] })
    score, train_history, trained_sequence_length = model_provider.train_model(labeled_jobs, params)
    settings_table.update({ 'key': sequence_key }, { 'key': sequence_key, 'value': trained_sequence_length }, upsert=True)
    return jsonify({ 'score': float(score), 'history': convert_history(train_history) })

@app.route('/api/months')
def get_months():
    num_months = request.args.get('numMonths', 6)
    months = job_provider.get_months(num_months)
    return jsonify([JobProvider.format_post(x) for x in months])

@app.route('/api/model/predict', methods=['POST'])
def predict_text():
    params = request.get_json()
    text = params['text']
    max_sequence_length = int(settings_table.find_one({ 'key': sequence_key })['value'])
    if len(text) > 0:
        prediction = model_provider.predict(text, max_sequence_length)
        return jsonify({ label_key: prediction })

@app.route('/api/model/predict/<int:parent_id>', methods=['GET'])
def predict_by_parent(parent_id):
    posts = job_provider.get_hiring_posts(parent_id)
    posts = [ JobProvider.format_post(x) for x in posts ]
    predictions = predict_bulk_inner(posts)
    preferred_ids = set([ p['id'] for p in predictions if p[label_key] == 1 ])
    return jsonify([ post for post in posts if post['id'] in preferred_ids ])

@app.route('/api/model/predictbulk', methods=['POST'])
def predict_bulk():
    jobs = request.get_json()
    return jsonify(predict_bulk_inner(jobs))

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
        oldest_id, newest_id = get_jobs_range(jobs_table)
        new_jobs = job_provider.get_next_post(historical_limit, oldest_id, newest_id)
        if new_jobs is not None:
            jobs_table.insert_many(new_jobs)
            return jsonify([JobProvider.format_post(x) for x in new_jobs])
        else:
            abort(404)

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def reactApp(path):
    if path != '' and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)


def convert_history(history):
    return { metric: convert_metric(history, metric) for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy'] }

def convert_metric(history, metric):
    return [ float(x) for x in history.get(metric, []) ]

def predict_bulk_inner(jobs):
    jobs = [ job for job in jobs if len(job.get('text', '')) > 0 ]
    max_sequence_length = int(settings_table.find_one({ 'key': sequence_key })['value'])
    if len(jobs) > 0:
        return model_provider.predict_bulk(jobs, max_sequence_length)
