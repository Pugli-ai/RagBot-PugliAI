from fastapi import FastAPI, Query, BackgroundTasks
from api import qa_run
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from api import qa_scraper
import traceback
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.responses import JSONResponse
from api import pinecone_functions

class ErrorResponse(BaseModel):
    detail: str


app = FastAPI()

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
    full_url: str

@app.post("/api/qa")
def qa_run_api(inputs: QA_Inputs):
    answer = qa_run.main(
        question=inputs.question,
        openai_api_key=inputs.gptkey,
        pinecone_index_name=inputs.kbid,
        chat_history_dict=inputs.chat_history_dict)

    return answer
    
# start_scrape api for scraping the url and saving the result into pinecone database
@app.post("/api/scrape")
async def scraper_api(inputs: Scraper_Inputs, background_tasks: BackgroundTasks):# Declare the variable as global so we can modify it

    if not pinecone_functions.is_api_key_valid(inputs.gptkey):
        return {"message": "Invalid Openai API key"}

    # Try to acquire the lock
    if qa_scraper.is_running:
        return {"message": f"Scraper is already running for {qa_scraper.current_url} website, please wait for it to finish."}
    else:

        background_tasks.add_task(qa_scraper.main, inputs.full_url, inputs.gptkey)
        return {"message": "Scrape started! Check scraping status API for progress."}

# generate_response api for checking the status of scraping process
@app.post("/api/scrape/status")
def scraper_status_api(inputs: Status_Inputs):
    status = qa_scraper.scraper_status(inputs.full_url)

    return status

# The api for deleting the index from pinecone database
@app.post("/api/scrape/delete")
def delete_index_api(inputs: Status_Inputs):
    status = qa_scraper.delete_index(inputs.full_url)

    return status

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    error_message = traceback.format_exc().splitlines()
    error_message = [x for x in error_message if x.strip()]
    error_message = error_message[-1]
    message = {"answer": "Error!", "source_url": None, "success": False, "error_message1": error_message }
    return JSONResponse(status_code=500, content=message)


"""
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
