from fastapi import FastAPI, BackgroundTasks
from api import qa_run
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from api import qa_scraper
import traceback
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.responses import JSONResponse
from api import pinecone_functions
import os
from queue import Queue
import asyncio
import time
from datetime import datetime
import pytz
from threading import Lock

lock = Lock()


class ErrorResponse(BaseModel):
    detail: str


app = FastAPI()
scraper_tree_queue = Queue()
scraper_single_queue = Queue()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Ciao Mondo"}

def is_url_in_queue(url):
    for item in list(scraper_tree_queue.queue):
        if url == item[0]:
            return True
    return False

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_worker())
    asyncio.create_task(single_queue_worker())  # Start the single queue worker


async def queue_worker():
    while True:
        if not scraper_tree_queue.empty():
            print("QUEUE ITEMS : ", list(scraper_tree_queue.queue))
            full_url, gptkey, namespace = scraper_tree_queue.get()
            await qa_scraper.main_async(full_url, gptkey, namespace) 
            scraper_tree_queue.task_done()
        await asyncio.sleep(2)  # Sleep for a short duration to prevent busy-waiting

async def single_queue_worker():
    while True:
        if not scraper_single_queue.empty():
            task = scraper_single_queue.get()
            # Process the task. You'll need to unpack and pass the arguments appropriately
            await qa_scraper.scrape_single_async(*task)
            scraper_single_queue.task_done()
        await asyncio.sleep(2)

######################################################################################
######################################## APIS ########################################
######################################################################################

### API for scraping a single page
class ScrapeSingleInputs(BaseModel):
    id: str # id of the single content
    content: str # content of the single content
    source: str # source of the single content (it can be url or a file name)
    type: str # “url | text”
    gptkey: str # "GPT-KEY"
    namespace: str # a generic alfa-num value (UUID etc.) passed on each query
    is_tree : str = "False" # if true, the content is a tree of contents (e.g. a websites)


@app.post("/api/scrape/single")
async def scrape_single_api(inputs: ScrapeSingleInputs, background_tasks: BackgroundTasks):
    print(f"{pinecone_functions.get_rome_time()} - /api/scrape/single is called")
    if not pinecone_functions.is_api_key_valid(inputs.gptkey):
        return {"message": "Invalid Openai API key"}
    else:
        # Add task to the queue instead of executing directly
        task = (inputs.id, inputs.content, inputs.source, inputs.type, inputs.gptkey, inputs.namespace, inputs.is_tree)
        scraper_single_queue.put(task)
        return {"message": "Scrape request added to queue! Check scraping status API for progress."}

## API for QA
class QA_Inputs(BaseModel):
    question: str
    gptkey :str
    namespace : str
    chat_history_dict: dict = {}
    model: str = "gpt-3.5-turbo" # or gpt-4

@app.post("/api/qa")
def qa_run_api(inputs: QA_Inputs):
    print(f"{pinecone_functions.get_rome_time()} - /api/qa is called")

    api_timer_start = time.time()
    answer = qa_run.main(
        question=inputs.question,
        openai_api_key=inputs.gptkey,
        namespace=inputs.namespace,
        model = inputs.model,
        chat_history_dict=inputs.chat_history_dict)
    
    api_timer_end = time.time()
    print("/api/qa TIME COST : ", api_timer_end - api_timer_start)
    return answer

## API for scraping a tree of pages
class Scraper_Inputs(BaseModel):
    full_url: str
    gptkey: str
    namespace: str

@app.post("/api/scrape/tree")
async def scraper_tree_api(inputs: Scraper_Inputs):
    print(f"{pinecone_functions.get_rome_time()} - /api/scrape/tree is called")

    if not pinecone_functions.is_api_key_valid(inputs.gptkey):
        return {"message": "Invalid Openai API key"}
    
    elif is_url_in_queue(inputs.full_url) or qa_scraper.current_url == inputs.full_url:
        return {"message": "This url is already in the queue!"}
    
    else:
        scraper_tree_queue.put((inputs.full_url, inputs.gptkey, inputs.namespace))
        return {"message": "Scrape request added to queue! Check scraping status API for progress."}

# Api for deleting id from namespace
class DeleteID_Inputs(BaseModel):
    id: str
    namespace: str


@app.post("/api/delete/id")
def delete_id_api(inputs: DeleteID_Inputs):
    print(f"{pinecone_functions.get_rome_time()} - /api/delete/id is called")

    status = qa_scraper.delete_with_id(inputs.id, inputs.namespace)
    return status

# Api for deleting namespace
class DeleteNamespace_Inputs(BaseModel):
    namespace: str

@app.post("/api/delete/namespace")
def delete_namespace_api(inputs: DeleteNamespace_Inputs):
    print(f"{pinecone_functions.get_rome_time()} - /api/delete/namespace is called")
    status = qa_scraper.delete_namespace(inputs.namespace)
    return status

# Api for listing namespace content
class ListNamespace_Inputs(BaseModel):
    namespace: str

@app.post("/api/list/namespace")
def list_namespace_api(inputs: ListNamespace_Inputs):
    print(f"{pinecone_functions.get_rome_time()} - /api/list/namespace is called")
    result = qa_scraper.list_namespace_content(inputs.namespace)
    return result

# Api for status of scraping process
class Status_Inputs(BaseModel):
    namespace_list: list
    id: str = None
    namespace: str = None

# generate_response api for checking the status of scraping process
@app.post("/api/scrape/status")
def scraper_status_api(inputs: Status_Inputs):
    with lock:  # Acquire the lock
        time.sleep(2)
        print(f"{pinecone_functions.get_rome_time()} - /api/scrape/status is called")
        # check if inputs.namespace_list is not empty:
        if inputs.namespace_list:
            status = qa_scraper.scraper_status_multi_pages(inputs.namespace_list, list(scraper_tree_queue.queue))
        else:
            status = qa_scraper.scraper_status_single(inputs.namespace, inputs.id)
    
    # The lock is automatically released here
    return status 


################################### EXCEPTION HANDLER ###################################

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    error_message = traceback.format_exc().splitlines()
    error_message = [x for x in error_message if x.strip()]
    error_message = error_message[-1]
    message = {"answer": "Error! (EXCEPTION HANDLER)", "source_url": None, "success": False, "error_message1": error_message }
    return JSONResponse(status_code=500, content=message)

