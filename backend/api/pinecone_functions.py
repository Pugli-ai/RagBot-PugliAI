import pinecone
import api_key #DEBUG
import os
from urllib.parse import urlparse

def init_pinecone():
    pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment="us-west1-gcp-free"  # find next to API key in console
    )

def create_index(index_name, dimention):
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(name=index_name, metric="cosine", dimension=dimention)

def retrieve_index(index_name):
    return pinecone.Index(index_name=index_name)

import hashlib

def hash_url(url):
    # Hash the URL using MD5
    md5 = hashlib.md5()
    md5.update(url.encode('utf-8'))
    hashed_url = md5.hexdigest()
    return hashed_url

def unhash_url(hashed_url):
    # To securely "unhash" a hashed URL, you cannot reverse the hashing process.
    # Instead, you can use a dictionary or database to store the mapping of hashed URLs to original URLs.
    # When you need to "unhash" a hashed URL, you can look up the original URL from the dictionary or database.
    # For simplicity, let's assume we use a global dictionary to store the mappings.

    # For demonstration purposes, I'm using a global dictionary. In practice, you may use a database.
    global url_mapping_dict

    if hashed_url in url_mapping_dict:
        return url_mapping_dict[hashed_url]
    else:
        return None

def get_domain_and_url(full_url):

    # make full url finish with / if not
    if not full_url.endswith("/"):
        full_url += "/"
    
    #make full_url start with https:// if not
    if not full_url.startswith("https://"):
        full_url = "https://" + full_url
    
    domain = urlparse(full_url).netloc # gethelp.tiledesk.com

    return full_url, domain
