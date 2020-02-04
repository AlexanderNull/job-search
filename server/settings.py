from dotenv import load_dotenv
from pathlib import Path
from os import getenv

def load_config():
    env_path = Path('.') / 'config' / '.env'
    load_dotenv(dotenv_path=env_path)

    # linkedin_credentials = getenv('LINKEDIN_CREDENTIALS')
    
    return {
        # 'LINKEDIN_CREDENTIALS': linkedin_credentials 
    }