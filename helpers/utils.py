import requests
import io
import os
from os import path

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith(
            'https://'):
        print(f'Fetching {str(url_or_path)}. \nThis might take a while... please wait.')
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

# Simple create path
def createPath(filepath):
    if path.exists(filepath) == False:
        os.makedirs(filepath)
        print(f'Made {filepath}')
    else:
        pass