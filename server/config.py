import json
import os

current_folder = os.path.dirname(__file__)
config_file_path = os.path.join(current_folder, '../config/pyserver.json')
with open(config_file_path) as json_file:
    config = json.load(json_file)
