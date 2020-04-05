from pymongo import MongoClient

from server.config import config
import server.db_tools as db_tools
from server.job_provider import JobProvider

client = MongoClient(config['mongo']['host'], config['mongo']['port'])
jobs_host = config['jobs_firebase_host']
throttle_group_size = 20
throttle_duration = 5
db_name = config['mongo']['embedding_data']['db_name']
table_name = config['mongo']['embedding_data']['table_name']

jobs_database = client[db_name]
jobs_for_embedding_table = jobs_database[table_name]

job_provider = JobProvider(jobs_host, throttle_group_size, throttle_duration)

oldest_id, newist_id = db_tools.get_jobs_range(jobs_for_embedding_table)

new_jobs = job_provider.get_next_post(6, oldest_id, newist_id)

if new_jobs is not None:
    jobs_for_embedding_table.insert_many(new_jobs)
    print(f"added {len(list(new_jobs))} new jobs")