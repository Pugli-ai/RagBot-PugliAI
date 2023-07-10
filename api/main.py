from fastapi import FastAPI, Query
from api import qa_run
import pandas as pd
import numpy as np

app = FastAPI()
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

@app.get("/")
async def root():
    return {"message": "Hello World"}
#examples
@app.get("/api")
async def genereteResponse():
    pass

@app.get("/api/{id}")
def home(id: int):
    return {"id": id}

@app.get("/qa")
def generate_response(question: str = Query(..., min_length=1)):
    answer = qa_run.answer_question(df, question=question, debug=False)
    print(answer)
    return answer

#http://localhost:8000/qa?question=How%20to%20connect%20Tiledesk%20with%20Telegram