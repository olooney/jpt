from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Union
import numpy as np
from util import human_join

from jpt import (
    index_search,
    alex,
    ken,
    larissa,
    david,
    james,
    brad,
    amy,
    mattea,
    load_jeopardy_index,
    load_jeopardy_dataset,
    clean_currency
)

contestant_map = {
    'ken': ken,
    'larissa': larissa,
    'david': david,
    'james': james,
    'brad': brad,
    'amy': amy,
    'mattea': mattea,
}
contestant_list = human_join(contestant_map.keys(), "or")

# Pydantic model for Jeopardy results
class JeopardyResult(BaseModel):
    category: str
    air_date: str
    question: str
    value: float
    answer: str
    round: str
    show_number: str
    has_link: bool
    distance: float

# Load the Jeopardy index and data
jeopardy_index = load_jeopardy_index()
jeopardy_data = load_jeopardy_dataset()

for question in jeopardy_data:
    question['value'] = clean_currency(question['value'])
    question['has_link'] = '<a href' in question['question']

def make_results_serializable(results):
    for result in results:
        result['value'] = float(result['value']) if isinstance(result['value'], (np.float32, np.float64)) else result['value']
        result['distance'] = float(result['distance']) if isinstance(result['distance'], (np.float32, np.float64)) else result['distance']
    return results

# Initialize FastAPI application
app = FastAPI()

# dev server convenience
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get('/search', response_model=List[JeopardyResult])
def search(
    query: str = Query(..., description="Query string to find semantically similar questions"), 
    k: int = Query(5, description="Number of results to return")) -> List[dict]:
    # validate args
    if k > 100: k = 100
        
    # Perform the search using the index_search function
    results = index_search(jeopardy_index, jeopardy_data, query, k=k)
    
    return make_results_serializable(results)

@app.get('/ask')
def ask(
    category: str = Query("GENERAL KNOWLEDGE", description="Column Category"),
    clue: str = Query(..., description="The prompt as read by Alex."),
    contestant: str = Query('ken', description=f"Contestant to answer: {contestant_list}"),
) -> str:
    contestant = contestant_map[contestant.lower()]
    return contestant(category, clue)

# Run this app using `uvicorn filename:app --reload`
