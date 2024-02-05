import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import openai
from time import sleep
import time
import traceback
import json
from datetime import datetime

# Attempt to import necessary modules from the API, if not available, import them directly.
try:
    from api import variables_db
    from api import pinecone_functions
except ImportError:
    import variables_db
    import pinecone_functions
import pinecone

import pandas as pd

if __name__ == "__main__":
    pinecone_functions.INDEX = pinecone_functions.retrieve_index()
    index_stats = pinecone_functions.INDEX.describe_index_stats()
    namespaces = list(index_stats['namespaces'].keys())

    for namespace in namespaces:
        print(f"Processing namespace {namespace}")
        # Write its ids to a csv
        result = pinecone_functions.INDEX.query(
            namespace=namespace,
            vector=[0.0] * index_stats['dimension'],
            top_k=1000,
            include_metadata=True
        )
        source_urls = set()
        for match in result["matches"]:
            id = match["id"]

            #source_url = match["metadata"]["text"].split("\n")[0].split(" ")[2]
            if id == variables_db.eof_index:
                continue
            source_urls.add(id)
        
        # Write to csv with namespace name
        df = pd.DataFrame(source_urls)
        df.to_csv(f"results/{namespace}.csv", index=False, header=False)