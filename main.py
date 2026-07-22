from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(
    title="Grounded Answer API",
    version="1.0"
)


class Chunk(BaseModel):
    chunk_id: str
    text: str


class Request(BaseModel):
    question: str
    chunks: List[Chunk]


class Response(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    answerable: bool


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


@app.get("/")
def home():
    return {"message": "Grounded Answer API is running"}


@app.post("/grounded-answer", response_model=Response)
def grounded_answer(req: Request):

    q_words = tokenize(req.question)

    best_chunk = None
    best_score = 0

    for chunk in req.chunks:
        chunk_words = tokenize(chunk.text)
        score = len(q_words & chunk_words)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return Response(
            answer="I don't know",
            citations=[],
            confidence=0.2,
            answerable=False
        )

    confidence = min(0.99, 0.5 + best_score * 0.1)

    return Response(
        answer=best_chunk.text,
        citations=[best_chunk.chunk_id],
        confidence=round(confidence, 2),
        answerable=True
    )
