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
import traceback
import json
from datetime import datetime
try: 
    from api import variables_db
    from api import pinecone_functions
except:
    import variables_db
    import pinecone_functions
import pinecone


# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []
        self.value_blacklist = [
            '.', '..', './', '../', '/', '#',
            'javascript:void(0);', 'javascript:void(0)', 
            'mailto:', 'tel:'
        ]


    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and value not in self.value_blacklist:
                    self.hyperlinks.append(value)

def get_hyperlinks(url):
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers)
        
        # If the response is not HTML, return an empty list
        if not response.headers.get('Content-Type').startswith("text/html"):
            return []
        
        html = response.text
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

def crawl_to_memory(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    texts= []
    # While the queue is not empty, continue crawling
    while queue:
        # Get the next URL from the queue
        url = queue.pop()
        #if url does not end with /, add it
        url = url.strip()
        if not url.endswith("/"):
            url += "/"
        
        print(url) # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        
        # Get the text from the URL using BeautifulSoup
        soup = BeautifulSoup(requests.get(url).text, "html.parser")

        # Get the text but remove the tags
        text = soup.get_text()

        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
        if ("You need to enable JavaScript to run this app." in text):
            print("Unable to parse page " + url + " due to JavaScript being required")
            continue
        # Otherwise, write the text to the file in the text directory
        text_name = 'text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt"
        text_name = text_name[11:-4].replace('-',' ').replace('_', ' ').replace('#update','')

        data = (url, text_name, text)
        texts.append(data)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

    return pd.DataFrame(texts, columns = ['url', 'title', 'text' ]).drop_duplicates(keep='last')


def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(url, text, tokenizer, max_tokens = 500):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append((url, ". ".join(chunk) + "."))
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def preprocess(df):
# Set the text column to be the raw text with the newlines removed
    df['text'] = df.title + ". " + remove_newlines(df.text)

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    #df.n_tokens.hist()

    max_tokens = 500

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():
        url = row[1]['url']
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:

            shortened += split_into_many(url, row[1]['text'], tokenizer=tokenizer, max_tokens=max_tokens)
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( (url, row[1]['text']) )

    df = pd.DataFrame(shortened, columns = ['url', 'text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    openai.api_key = variables_db.OPENAI_API_KEY

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    return df

def convertdf2_pineconetype(df):
    """
    Convert a pandas dataframe to a metadata type
    """
    datas = []
    for row in df.itertuples():
        metadata = {'url': row.url, 'n_tokens': row.n_tokens, 'text': row.text}
        data = {'id': row.url, 'values':row.embeddings, 'metadata': metadata}
        datas.append(data)
    return pd.DataFrame(datas)

def crawl_deghi():
    url = "https://www.deghi.it/supporto"  # Replace this with the URL of the webpage you want to fetch

    response = requests.get(url)

    source_code = response.text

    # Parse the HTML source code using BeautifulSoup
    soup = BeautifulSoup(source_code, "html.parser")

    # Find all the div elements with the specified class name
    divs_with_class = soup.find_all("div", class_="card border-no mb-5")

    # Create lists to store the data
    headers = []
    bodies = []

    # Process the found div elements and extract header and body content
    for div in divs_with_class:
        header = div.find("div", class_="card-header").text.strip()
        body = div.find("div", class_="card-body").text.strip()
        body = header + "\n\n" + body
        # remove non ascii from header
        header = header.encode("ascii", "ignore").decode()
        headers.append(header)
        bodies.append(body)

    # Create a Pandas DataFrame

    return  pd.DataFrame({"url": headers, "title": headers, "text": bodies})

def scraper_status(full_url):
    try:
        variables_db.PINECONE_INDEX_NAME = pinecone_functions.url_to_index_name(full_url)

        pinecone_functions.init_pinecone(variables_db.PINECONE_API_KEY, variables_db.PINECONE_API_KEY_ZONE)



        #check if there is an index with that name in pinecone
        if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
            result = pinecone_functions.INDEX.fetch(ids=[variables_db.eof_index])
            if len(result['vectors']) <1:
                message = f"Crawling is started but not finished yet for {full_url}"
            else:
                message = f"Crawling is finished for {full_url}"
            
        else:
            message = f"Database is not created yet for {full_url}, please wait a few minutes and try again"
    except:
        message = f"Database is not created yet for {full_url}, please wait a few minutes and try again"
    return message

def main(full_url: str, gptkey:str):
    variables_db.OPENAI_API_KEY = gptkey
    full_url, domain = pinecone_functions.get_domain_and_url(full_url)
    
    pinecone_functions.init_pinecone(variables_db.PINECONE_API_KEY, variables_db.PINECONE_API_KEY_ZONE)
    index_name = pinecone_functions.url_to_index_name(full_url)
    print('index name: ', index_name)
    variables_db.PINECONE_INDEX_NAME = index_name

    if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
        print("deleting the eof row")
        pinecone_functions.INDEX = pinecone_functions.retrieve_index()
        pinecone_functions.INDEX.delete(ids=[variables_db.eof_index])

    print('Crawling...')
    
    if full_url=="https://www.deghi.it/supporto/":
        df = crawl_deghi()
    else:
        df = crawl_to_memory(full_url)

    eof_row = {'url': variables_db.eof_index, 'title': variables_db.eof_index, 'text': variables_db.eof_index}
    df = pd.concat([df, pd.DataFrame([eof_row])], ignore_index=True)

    print('Crawling completed.')
    
    df = preprocess(df)

    dimention = len(df.iloc[0]['embeddings'])

    print('creating index...')
    pinecone_functions.create_index(dimention)

    print('index created, retrieving index...')
    sleep(2)
    pinecone_functions.INDEX = pinecone_functions.retrieve_index()

    print('Uploading data to Pinecone...')
    df = convertdf2_pineconetype(df)
    pinecone_functions.INDEX.upsert_from_dataframe(df, batch_size=2)

    print("Data upsert completed.")

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

if __name__ == "__main__":
    # Define root domain to crawl
    full_url = "https://gethelp.tiledesk.com/"
    #full_url = "https://www.deghi.it/supporto/"
    main(full_url, variables_db.OPENAI_API_KEY)