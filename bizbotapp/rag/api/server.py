from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag.bot import BizBot
from rag.telemetry import log_query

app = FastAPI()
bot = BizBot()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_bot(request: QueryRequest):
    query = request.query
    answer = bot.answer(query)
    log_query(query, "[context handled internally]", answer)
    return {"answer": answer}

@app.get("/")
def root():
    return {"status": "BizBot RAG API is running."}