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
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from urllib.parse import urlencode

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager

is_outsourceapi= False
is_selenium = True

is_running = False
current_url = None

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

class HyperlinkParser(HTMLParser):
    def __init__(self) -> None:
        """
        Initialize the HyperlinkParser.
        """
        super().__init__()
        self.hyperlinks = []
        self.value_blacklist = [
            '.', '..', './', '../', '/', '#',
            'javascript:void(0);', 'javascript:void(0)', 
            'mailto:', 'tel:'
        ]

    def handle_starttag(self, tag: str, attrs: tuple) -> None:
        """
        Handle the start tag of an HTML element.

        Args:
            tag (str): The tag name.
            attrs (tuple): Attributes of the tag.
        """
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and value not in self.value_blacklist:
                    self.hyperlinks.append(value)

def get_hyperlinks(url: str) -> list:
    """
    Fetch hyperlinks from a given URL.

    Args:
        url (str): The URL to fetch hyperlinks from.

    Returns:
        list: List of hyperlinks.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    global is_outsourceapi
    try:
        if is_outsourceapi:
            response = requests.get('http://api.scraperapi.com/', params=urlencode({'api_key': "3555494125444f7cb3f561e2062d80c1", 'url': url}))
        else:
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

def get_hyperlinks_with_selenium(driver: webdriver) -> list:
    """
    Fetch hyperlinks from a given URL using Selenium WebDriver.

    Args:
        driver (webdriver): The Selenium WebDriver instance.

    Returns:
        list: List of hyperlinks.
    """
    try:
        # Get the current URL and page source from the driver
        url = driver.current_url
        html = driver.page_source

        # If the URL is not HTML, return an empty list
        # Note: Selenium usually deals with HTML, so this check might be redundant
        if not "text/html" in driver.execute_script("return document.contentType"):
            return []
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

def get_domain_hyperlinks_with_selenium(local_domain: str, driver: webdriver) -> list:
    """
    Fetch hyperlinks from a URL that are within the same domain.

    Args:
        local_domain (str): The domain to filter links by.
        driver (webdriver): The Selenium WebDriver instance.

    Returns:
        list: List of hyperlinks within the domain.
    """
    clean_links = []
    for link in set(get_hyperlinks_with_selenium(driver)):
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

def get_domain_hyperlinks(local_domain: str, url: str) -> list:
    """
    Fetch hyperlinks from a URL that are within the same domain.

    Args:
        local_domain (str): The domain to filter links by.
        url (str): The URL to fetch hyperlinks from.

    Returns:
        list: List of hyperlinks within the domain.
    """
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

def crawl_to_memory(url: str) -> pd.DataFrame:
    """
    Crawl a website and store its content in memory.

    Args:
        url (str): The URL of the website to crawl.

    Returns:
        pd.DataFrame: DataFrame containing the crawled content.
    """
    global is_outsourceapi, is_selenium
    print("SELENIUM TRY TO INIT")
    if is_selenium:
        # Set up the webdriver
        # Set up the webdriver
        firefox_options = FirefoxOptions()
        firefox_options.add_argument('--headless')  # Uncomment to run headless
        firefox_options.add_argument("--log-level=3")

        driver = webdriver.Firefox(service=Service(executable_path="./geckodriver"), options=firefox_options)
        print("######################")
        print(driver)
        print("######################")
        sleep(1)
        # Navigate to the webpage
        driver.get(url)

    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    texts = []
    # While the queue is not empty, continue crawling
    if is_selenium:
        sleep(10)

    while queue and len(texts) < 10:
        # Get the next URL from the queue
        url = queue.pop()
        # If url does not end with /, add it
        url = url.strip()
        
        if not url.endswith("/"):
            url += "/"
        url = url.replace('%2F%2F%2F', '')

        print(url)  # For debugging and to see the progress
        #url = "https://ecobaby.it/shop/primi-giochi/840-6065-https-ecobaby-it-shop-primi-giochi-840-6066-Pinguino-sempre-in-piedi-html.html#/4565-colore-grigio/"
        
        # Get the text from the URL using BeautifulSoup
        if is_outsourceapi:
           text =  requests.get('http://api.scraperapi.com/', params=urlencode({'api_key': "3555494125444f7cb3f561e2062d80c1", 'url': url})).text
        elif is_selenium:
            try:
                driver.get(url)
                sleep(1)
                text = driver.find_element(By.TAG_NAME, 'body').text
            except:
                print("Unable to parse page: " + url)
                continue  
        else:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            # Get the text but remove the tags
            text = soup.get_text()

        clean_url = url[:-1] if url.endswith("/") else url
        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
        if ("You need to enable JavaScript to run this app." in text):
            print("Unable to parse page " + url + " due to JavaScript being required")
            continue
        elif any(clean_url.endswith(ext) for ext in variables_db.avoid_extensions):
            continue

        # Otherwise, write the text to the file in the text directory
        text_name = 'text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt"
        text_name = text_name[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', '')
        #text = "source url: "+ url + "\n" + text
        data = (url, text_name, text)
        texts.append(data)
        if is_selenium:
            # Get the hyperlinks from the URL and add them to the queue
            for link in get_domain_hyperlinks_with_selenium(local_domain, driver): # is_selenium get_domain_hyperlinks_with_selenium or get_domain_hyperlinks
                link = link.replace('%2F%2F%2F', '')
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
        else:  
            # Get the hyperlinks from the URL and add them to the queue
            for link in get_domain_hyperlinks(local_domain, url): # is_selenium get_domain_hyperlinks_with_selenium or get_domain_hyperlinks
                link = link.replace('%2F%2F%2F', '')
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
    if is_selenium:
        driver.quit()

    return pd.DataFrame(texts, columns=['url', 'title', 'text']).drop_duplicates(keep='last')

def remove_newlines(serie: pd.Series) -> pd.Series:
    """
    Remove newlines and extra spaces from a pandas Series.

    Args:
        serie (pd.Series): Series containing text data.

    Returns:
        pd.Series: Cleaned series.
    """
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.apply(lambda x: re.sub(' +', ' ', x))
    return serie

def split_into_many(url: str, text: str, tokenizer, max_tokens: int) -> list:
    """
    Split text into chunks based on a maximum number of tokens.

    Args:
        url (str): The URL associated with the text.
        text (str): The text to be split.
        tokenizer: The tokenizer to use.
        max_tokens (int, optional): Maximum tokens per chunk.

    Returns:
        list: List of text chunks.
    """
    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    # token of url
    title = "source url: " + url + " "
    title_n_tokens = len(tokenizer.encode(" " + title))
    max_tokens = max_tokens - title_n_tokens

    chunks = []
    tokens_so_far = 0
    chunk = []
    counter=0

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        
        if tokens_so_far + token > max_tokens:
            final_text = ". ".join(chunk) + "."
            final_text = title + final_text
            url_with_id = url +"#"+str(counter)
            chunks.append((url_with_id, final_text))
            chunk = []
            tokens_so_far = 0
            counter+=1

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame containing website content.

    Args:
        df (pd.DataFrame): DataFrame containing website content.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Set the text column to be the raw text with the newlines removed
    df['text'] = remove_newlines(df.text)

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    max_tokens = 1000

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
            text= "source url: " + url + " " + row[1]['text']
            shortened.append((url, text))

    df = pd.DataFrame(shortened, columns=['url', 'text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    openai.api_key = variables_db.OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = openai.api_key
    print("embedding started")
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    print("embedding ended")
    return df

def convertdf2_pineconetype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a pandas dataframe to a metadata type suitable for Pinecone.

    Args:
        df (pd.DataFrame): DataFrame to convert.

    Returns:
        pd.DataFrame: Converted DataFrame.
    """
    datas = []
    for row in df.itertuples():
        metadata = {'url': row.url, 'n_tokens': row.n_tokens, 'text': row.text}
        data = {'id': row.url.encode("ascii", "ignore").decode(), 'values': row.embeddings, 'metadata': metadata}
        datas.append(data)
    return pd.DataFrame(datas)

def crawl_deghi() -> pd.DataFrame:
    """
    Temporary method for scraping the deghi website.

    Returns:
        pd.DataFrame: DataFrame containing scraped content.
    """
    url = "https://www.deghi.it/supporto"

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
        #body = "source url: "+ header + "\n" + body
        # Remove non-ASCII characters from header
        header = header.encode("ascii", "ignore").decode()
        headers.append(header)
        bodies.append(body)

    ### Begining of Temporary section
    temp_list = ["https://www.deghi.it/ombrellone-da-giardino-3x3-m-palo-laterale-telo-ecr-tenerife",
                 "https://www.deghi.it/gazebo-3x36-in-alluminio-e-policarbonato-grigio-sawara",
                 "https://www.deghi.it/ombrellone-da-giardino-3x2-m-palo-centrale-telo-tortora-robin",
                 "https://www.deghi.it/poltrona-da-giardino-in-textilene-e-acciaio-marrone-pados",
                 "https://www.deghi.it/lettino-imbottito-reclinabile-in-alluminio-antracite-azimut",
                 "https://www.deghi.it/set-2-sedie-pieghevoli-da-giardino-in-legno-di-teak-louis",
                 "https://www.deghi.it/tavolo-allungabile-da-giardino-160-240-cm-in-alluminio-tortora-carioca",
                 "https://www.deghi.it/tavolo-ovale-allungabile-150-200-cm-in-legno-di-acacia-paja",
                 "https://www.deghi.it/tavolino-da-giardino-70x50-cm-in-acciaio-marrone-pados",
                 "https://www.deghi.it/set-relax-5-pezzi-per-composizione-libera-divano-chaise-longue-grigio-con-tavolino"]
    

    for url in temp_list:
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        temp_url_content = soup.get_text()
        temp_url_content = url + "\n\n" + temp_url_content
        headers.append(url)
        bodies.append(temp_url_content)


    ### End of temporary section
    # Create a Pandas DataFrame
    return pd.DataFrame({"url": headers, "title": headers, "text": bodies})

def scraper_status(full_url: str) -> dict:
    """
    Check the status of the scraper for a given URL.

    Args:
        full_url (str): URL to check the scraper status for.

    Returns:
        dict: Status message and code. 0-> not started, 1-> started, 2-> finished
    """
    global is_running, current_url
    try:
        pinecone_namespace = pinecone_functions.url_to_namespace(full_url)
        
        if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
            result = pinecone_functions.INDEX.fetch(ids=[variables_db.eof_index], namespace=pinecone_namespace)

            if is_running and current_url == full_url:
                message = f"Crawling is started but not finished yet for {full_url}"
                status_code = 1
            elif len(result['vectors']) > 0:
                status_code = 2
                message = f"Crawling is finished for {full_url}"
            else:
                status_code = 0
                message = f"Database is not created yet for {full_url}, please wait a few minutes and try again"
        else:
            status_code = 0
            message = f"Database is not created yet for {full_url}, please wait a few minutes and try again"
    
        response = {"status_message" : message, "status_code": status_code}
        return response
    
    except:
        traceback.print_exc()
        status_code = 0
        message = traceback.format_exc()
    
    return {"status_message" : message, "status_code": status_code}

def delete_namespace(full_url: str) -> None:
    """delete pinecone index if exists

    Args:
        full_url (str): _description_
    """
    pinecone_namespace = pinecone_functions.url_to_namespace(full_url)

    # Check if there is an index with that name in pinecone
    if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
        pinecone_functions.INDEX = pinecone_functions.retrieve_index()
        index_stats = pinecone_functions.INDEX.describe_index_stats()
        namespaces = list(index_stats['namespaces'].keys())
        if pinecone_namespace in namespaces:
            pinecone_functions.INDEX.delete(delete_all=True, namespace=pinecone_namespace)
            response = {"success": True, "message": f"{full_url} is deleted from database"}
        else:
            response = {"success": False, "message": f"{full_url} is not in database"}
    else:
        response = {"success": False, "message": f"{full_url} is not in database"}
    return response
def main(full_url: str, gptkey: str) -> None:
    """
    Main function to start the scraping process.

    Args:
        full_url (str): URL to scrape.
        gptkey (str): OpenAI API key.
    """
    global is_running, current_url
    try:
        is_running = True
        current_url = full_url
        timer_start = time.time()
        variables_db.OPENAI_API_KEY = gptkey
        full_url, domain = pinecone_functions.get_domain_and_url(full_url)

        pinecone_namespace = pinecone_functions.url_to_namespace(full_url)
        print('pinecone_namespace: ', pinecone_namespace)
        #variables_db.PINECONE_INDEX_NAME = index_name

        if variables_db.PINECONE_INDEX_NAME in pinecone.list_indexes():
            
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
            index_stats = pinecone_functions.INDEX.describe_index_stats()
            namespaces = list(index_stats['namespaces'].keys())

            if pinecone_namespace in namespaces:
                print("If eof row exists, deleting the eof row on namespace: ", pinecone_namespace)
                pinecone_functions.INDEX.delete(ids=[variables_db.eof_index], namespace=pinecone_namespace)

        timer_end = time.time()
        print(f"Time to initialize pinecone: {format_time(timer_end - timer_start)}")

        print('Crawling...')
        timer_start = time.time()
        if full_url == "https://www.deghi.it/supporto/":
            df = crawl_deghi()
        else:
            df = crawl_to_memory(full_url)

        eof_row = {'url': variables_db.eof_index, 'title': variables_db.eof_index, 'text': variables_db.eof_index}
        df = pd.concat([df, pd.DataFrame([eof_row])], ignore_index=True)

        print('Crawling completed.')
        timer_end = time.time()
        print(f"Time to crawl: {format_time(timer_end - timer_start)}")
        timer_start = time.time()
        df = preprocess(df)

        dimension = len(df.iloc[0]['embeddings'])
        timer_end = time.time()
        print(f"Time to preprocess: {format_time(timer_end - timer_start)}")

        print('creating index...')
        timer_start = time.time()
        pinecone_functions.create_index(dimension)

        print('index created, retrieving index...')
        sleep(2)
        pinecone_functions.INDEX = pinecone_functions.retrieve_index()

        print('Uploading data to Pinecone...')
        df = convertdf2_pineconetype(df)
        pinecone_functions.INDEX.upsert_from_dataframe(df, batch_size=2, namespace=pinecone_namespace)
        is_running = False
    except:
        is_running = False
        #traceback.print_exc()
        print(traceback.format_exc())

    print("Data upsert completed.")
    timer_end = time.time()
    print(f"Time to upload data to pinecone: {format_time(timer_end - timer_start)}")

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    time_str = ""
    if days > 0:
        time_str += f"{days}d "
    if hours > 0:
        time_str += f"{hours}h "
    if minutes > 0:
        time_str += f"{minutes}min "
    if seconds > 0:
        time_str += f"{seconds}sec"
    else:
        time_str += "0sec"

    return time_str

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

if __name__ == "__main__":
    # Define root domain to crawl
    #full_url = "https://gethelp.tiledesk.com/"
    #full_url = "https://www.deghi.it/supporto/"
    #full_url = "https://knowledge.webafrica.co.za"
    full_url = "https://aulab.it/"
    #full_url = "https://ecobaby.it/"
    #full_url = "https://lineaamica.gov.it/"
    #full_url = "http://cairorcsmedia.it/"
    #full_url = "https://www.sace.it/"
    #full_url = "https://www.ucl.ac.uk/"
    #full_url = "http://iostudiocongeco.it/"
    #full_url = "https://www.postpickr.com/"
    #full_url = "https://www.loloballerina.com/"
    main(full_url, variables_db.OPENAI_API_KEY)