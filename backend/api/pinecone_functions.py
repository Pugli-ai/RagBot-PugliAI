import pinecone
from api import variables_db
import os
from urllib.parse import urlparse

INDEX= None

def init_pinecone():
    pinecone.init(
    api_key=variables_db.PINECONE_API_KEY,
    environment="us-west1-gcp-free"  # find next to API key in console
    )

def create_index(dimention):
    if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
        pinecone.delete_index(variables_db.PINECONE_INDEX_NAME)
    pinecone.create_index(name=variables_db.PINECONE_INDEX_NAME, metric="cosine", dimension=dimention)

def retrieve_index():
    return pinecone.Index(index_name=variables_db.PINECONE_INDEX_NAME)

import re

def url_to_index_name(url):
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
    if not full_url.startswith("https://"):
        full_url = "https://" + full_url
    
    domain = urlparse(full_url).netloc # gethelp.tiledesk.com

    return full_url, domain


