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



class ErrorResponse(BaseModel):
    detail: str


app = FastAPI()
task_queue = Queue()

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

class QA_Inputs(BaseModel):
    question: str
    gptkey :str
    kbid : str
    chat_history_dict: dict = {}

class Scraper_Inputs(BaseModel):
    full_url: str
    gptkey: str

class Status_Inputs(BaseModel):
    url_list: list

class Delete_Inputs(BaseModel):
    full_url: str

def is_url_in_queue(url):
    for item in list(task_queue.queue):
        if url == item[0]:
            return True
    return False

@app.post("/api/qa")
def qa_run_api(inputs: QA_Inputs):
    answer = qa_run.main(
        question=inputs.question,
        openai_api_key=inputs.gptkey,
        full_url=inputs.kbid,
        chat_history_dict=inputs.chat_history_dict)

    return answer

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_worker())

async def queue_worker():
    while True:
        if not task_queue.empty():
            print("QUEUE ITEMS : ", list(task_queue.queue))
            full_url, gptkey = task_queue.get()
            await qa_scraper.main_async(full_url, gptkey) 
            task_queue.task_done()
        await asyncio.sleep(1)  # Sleep for a short duration to prevent busy-waiting

######################################## APIS ########################################
@app.post("/api/scrape")
async def scraper_api(inputs: Scraper_Inputs):

    if not pinecone_functions.is_api_key_valid(inputs.gptkey):
        return {"message": "Invalid Openai API key"}
    
    elif is_url_in_queue(inputs.full_url) or qa_scraper.current_url == inputs.full_url:
        return {"message": "This url is already in the queue!"}
    
    else:
        task_queue.put((inputs.full_url, inputs.gptkey))
        return {"message": "Scrape request added to queue! Check scraping status API for progress."}


# generate_response api for checking the status of scraping process
@app.post("/api/scrape/status")
def scraper_status_api(inputs: Status_Inputs):
    status = qa_scraper.scraper_status_multi_pages(inputs.url_list, list(task_queue.queue))
    return status

# The api for deleting the index from pinecone database
@app.post("/api/scrape/delete")
def delete_index_api(inputs: Delete_Inputs):
    status = qa_scraper.delete_namespace(inputs.full_url)

    return status

@app.post("/api/scrape/list_queue")
def list_queue_api():
    return list(task_queue.queue)

################################### EXCEPTION HANDLER ###################################

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    error_message = traceback.format_exc().splitlines()
    error_message = [x for x in error_message if x.strip()]
    error_message = error_message[-1]
    message = {"answer": "Error! (EXCEPTION HANDLER)", "source_url": None, "success": False, "error_message1": error_message }
    return JSONResponse(status_code=500, content=message)


"""

# start_scrape api for scraping the url and saving the result into pinecone database
@app.post("/api/pwd")
async def pwd():
    pwd = ""
    pwd_items =""
    parent_items = ""
    api_items=""
    pwd = os.getcwd()
    try:
        pwd= os.getcwd()
        pwd_items = os.listdir()
    except:
        pass
    try:
        parent_items = os.listdir(os.path.dirname(os.getcwd()))
    except:
        pass
    try:
        api_items= os.listdir("__pycache__")
    except:
        pass
    return {"pwd": pwd, "pwd_items": pwd_items, "parent_items": parent_items, "api_items": api_items}


    
@app.get("/qa")
def generate_response(question: str = Query(..., min_length=1)):
    answer = qa_run.answer_question(df, question=question, debug=False)
    print(answer)
    return answer

#http://localhost:8000/qa?question=How%20to%20connect%20Tiledesk%20with%20Telegram


@app.post("/api/scrape")
async def start_scrape(url: Scraper_Inputs):
    qa_scraper.main(url.full_url)
    return {"message": "Scrape finished!"}

"""
