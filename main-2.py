from transformers import pipeline
from fastapi import FastAPI 
from typing import Union
import pytesseract




app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/answerFromText")
#/{{question}:{context}}")
def read_item():
    #print(question , context)
    nlp_qa = pipeline("question-answering",model="distilbert-base-cased-distilled-squad")
    return nlp_qa({
    'question': "what is ml",
    'context': "ml is anything"
})

@app.get("/answerFromImage/{{image}:{question}}")
async def read_item( image,question ):
   
  
    nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
    )

    return nlp(image,question)

@app.get("/summarize/{{article}:{maxLength}}")
async def summarize( article,maxLength=130 ):
   
  
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(article, max_length=maxLength, min_length=30, do_sample=False)
