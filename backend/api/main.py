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
    return {"message": "Hello World"}

class QA_Inputs(BaseModel):
    question: str
    gptkey :str
    kbid : str
    chat_history_dict: dict

class Scraper_Inputs(BaseModel):
    full_url: str
    gptkey: str

class Status_Inputs(BaseModel):
    full_url: str

@app.post("/api/qa")
def generate_response(inputs: QA_Inputs):
    answer = qa_run.main(
        question=inputs.question,
        openai_api_key=inputs.gptkey,
        pinecone_index_name=inputs.kbid,
        chat_history_dict=inputs.chat_history_dict)

    return answer
    
# Make the start_scrape endpoint asynchronous
@app.post("/api/scrape")
async def start_scrape(inputs: Scraper_Inputs, background_tasks: BackgroundTasks):
    if not pinecone_functions.is_api_key_valid(inputs.gptkey):
        return {"message": "Invalid Openai API key"}
    # Run qa_scraper.main in the background using BackgroundTasks
    background_tasks.add_task(qa_scraper.main, inputs.full_url, inputs.gptkey)
    return {"message": "Scrape started! Check logs for progress."}

@app.post("/api/scrape/status")
def generate_response(inputs: Status_Inputs):
    status = qa_scraper.scraper_status(inputs.full_url)

    return {"status": status}

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
