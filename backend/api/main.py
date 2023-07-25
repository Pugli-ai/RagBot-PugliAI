from fastapi import FastAPI, Query
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
    
@app.post("/scrape/start")
async def start_scrape(url: Url):
    qa_scraper.main(url.full_url)
    return {"message": "Scrape finished!"}

"""
@app.get("/qa")
def generate_response(question: str = Query(..., min_length=1)):
    answer = qa_run.answer_question(df, question=question, debug=False)
    print(answer)
    return answer

#http://localhost:8000/qa?question=How%20to%20connect%20Tiledesk%20with%20Telegram
"""
