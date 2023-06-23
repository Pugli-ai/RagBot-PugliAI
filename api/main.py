from fastapi import FastAPI

app = FastAPI()

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
