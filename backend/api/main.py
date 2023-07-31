from fastapi import FastAPI, Query, BackgroundTasks

from api import qa_run
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from api import qa_scraper

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


class Response(BaseModel):
    question: str
    gptkey :str
    kbid : str

@app.post("/api/qa")
def generate_response(response: Response):
    answer = qa_run.main(question=response.question, openai_api_key=response.gptkey, pinecone_index_name=response.kbid)
    print(answer)
    return answer

class Url(BaseModel):
    full_url: str
    
# Make the start_scrape endpoint asynchronous
@app.post("/api/scrape")
async def start_scrape(url: Url, background_tasks: BackgroundTasks):
    # Run qa_scraper.main in the background using BackgroundTasks
    background_tasks.add_task(qa_scraper.main, url.full_url)
    return {"message": "Scrape started! Check logs for progress."}

@app.post("/api/scrape/status")
def generate_response(url: Url):
    status = qa_scraper.scraper_status(url.full_url)

    return {"status": status}

"""
@app.get("/qa")
def generate_response(question: str = Query(..., min_length=1)):
    answer = qa_run.answer_question(df, question=question, debug=False)
    print(answer)
    return answer

#http://localhost:8000/qa?question=How%20to%20connect%20Tiledesk%20with%20Telegram


@app.post("/api/scrape")
async def start_scrape(url: Url):
    qa_scraper.main(url.full_url)
    return {"message": "Scrape finished!"}

"""
