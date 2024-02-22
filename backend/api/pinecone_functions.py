import pinecone
import os
from urllib.parse import urlparse
from openai import OpenAI


try:
    from api import variables_db
except:
    import variables_db

import traceback

from datetime import datetime
import pytz
    
INDEX= None
is_pinecone_initialized = False
client = OpenAI(api_key=variables_db.OPENAI_API_KEY)

def init_pinecone():
    api_key = variables_db.PINECONE_API_KEY
    api_key_zone = variables_db.PINECONE_API_KEY_ZONE
    pinecone.init(
    api_key=api_key,
    environment=api_key_zone  # find next to API key in console
    )
    
if not is_pinecone_initialized:
    init_pinecone()
    is_pinecone_initialized = True

def is_api_key_valid(api_key):
    try:
        client.models.list()
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False

def is_db_exists():
    """ Check if a database exists
    """
    if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
        return True
    else:
        return False


def create_index(dimention):
    if is_db_exists():
        print(f"{variables_db.PINECONE_INDEX_NAME} already exists. Skipping creation")
    else:
        print(f"Creating {variables_db.PINECONE_INDEX_NAME} index")
        pinecone.create_index(name=variables_db.PINECONE_INDEX_NAME, metric="cosine", dimension=dimention)


def retrieve_index():
    return pinecone.Index(index_name=variables_db.PINECONE_INDEX_NAME)

import re

def url_to_namespace(url):
    # Remove 'https://' or 'http://'
    url = re.sub(r'^https?://', '', url)

    # Remove trailing slashes
    url = url.rstrip('/')

    # Replace non-alphanumeric characters with '-'
    url = re.sub(r'[^a-zA-Z0-9]+', '-', url)

    # Remove leading '-' if present
    url = url.lstrip('-')

    # Remove leading and trailing '-' if present
    url = url.strip('-')

    # Convert to lowercase
    url = url.lower()

    return url

def get_domain_and_url(full_url):

    # make full url finish with / if not
    if not full_url.endswith("/"):
        full_url += "/"
    
    #make full_url start with https:// if not
    if not full_url.startswith("https://") and not full_url.startswith("http://"):
        full_url = "https://" + full_url
    
    domain = urlparse(full_url).netloc # gethelp.tiledesk.com

    return full_url, domain


def get_rome_time():
    """Returns the current date and time in Rome timezone as a string."""
    rome_tz = pytz.timezone('Europe/Rome')
    rome_time = datetime.now(rome_tz)
    return rome_time.strftime("%Y-%m-%d %H:%M:%S")