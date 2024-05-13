import json
import numpy as np
import pandas as pd
import logging
from typing import List, Union, Any, Callable, Tuple
from dataclasses import dataclass
import re
import math
import os
import random

# import ollama # deferred
import openai
import boto3
import tenacity
import faiss
import joblib

from util import Credentials, ws, TemporarySeed


# define agent types for type annotation
PlayerAgent = Callable[[str, str], str]
HostAgent = Callable[[str, str, str, str], str]

logger = logging.getLogger(__name__)

CONTESTANT_NAME_MAP = { 
    'ken': 'Ken Jennings\n(gpt-4-turbo)',
    'larissa': 'Larissa Kelly\n(llama3:8b)',
    'david': 'David Madden\n(llama2:7b)',
    'james': 'James Holzhauer\n(gpt-3.5-turbo)', 
    'brad': 'Brad Rutter\n(llama3:70b)',
    'amy': 'Amy Schneider\n(gpt-3.5-fine-tuned)',
    'mattea': 'Mattea Roach\n(llama3:8b + RAG)',
}

JEOPARDY_DATA_DIR = r'D:\Dropbox\data\jeopardy'
if not os.path.isdir(JEOPARDY_DATA_DIR):
    JEOPARDY_DATA_DIR = r'C:\Users\oloon\Dropbox\data\jeopardy'

# shared OpenAI client
openai_credentials_filename = os.path.join(os.path.expanduser('~'), '.openai', 'credentials.yaml')
openai_credentials = Credentials.load(openai_credentials_filename)
client = openai.OpenAI(
    organization=openai_credentials.organization, 
    api_key=openai_credentials.api_key
)

jeopardy_chunk_template = '''JEOPARDY QUESTION:
category: {category!r}
clue: {question}
correct response: {answer!r}'''

jeopardy_question_template = '''
CATEGORY: {category}
{clue}
'''

system_messages = [
    {"role": "system", "content": "You are a contestant on Jeopardy. Each prompt has both the category (column header) and Jeopardy clue; answer in the form of a question."},
    {"role": "user", "content": "CATEGORY: THE BIG APPLE\nThere's an annual footrace up its 86 flights of stairs"},
    {"role": "assistant", "content": "What is the Empire State Building?"}
]


def full_name(contestant: PlayerAgent) -> str:
    '''given a reference to an agent, returns their full, human readable name
    (including their underlying algorithm.)
    '''
    if isinstance(contestant, str):
        return CONTESTANT_NAME_MAP[contestant]
    elif hasattr(contestant, '__name__'):
        return CONTESTANT_NAME_MAP[contestant.__name__]
    else:
        raise ValueError(f"Unknown contestant {contestant!r}")


def format_llama3_prompt(messages: List[dict]) -> str:
    '''AWS Bedrock accepts only a single string in raw format; we need to format
    messages using their special tags. Llama3 uses a different format from llama2.
    '''
    prompt = '<|begin_of_text|>'
    for message in messages:
        role = message['role']
        content = message['content']
        assert role in ['system', 'user', 'assistant']
        prompt += f'<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>'
    prompt += '<|start_header_id|>assistant<|end_header_id|>'
    return prompt

# reusable decorator to implement basic retry logic. We make up to three
# attempts with exponential backoff. 
retry_decorator = tenacity.retry(
    wait=tenacity.wait_exponential(min=0.1, max=2),
    stop=tenacity.stop_after_attempt(3), # because 4 is too many and 2 isn't enough.
    after=tenacity.after_log(logger, logging.ERROR),
    reraise=True)


@retry_decorator
def gpt(prompt: str) -> str:
    '''Generic call to OpenAI's API.'''

    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    content = chat_response.choices[0].message.content
    return content


@retry_decorator
def ken(category: str, clue: str) -> str:
    '''PlayerAgent for GPT-4. Calls OpenAI.'''
    
    prompt = jeopardy_question_template.format(**locals())
    chat_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=system_messages+[
            {"role": "user", "content": prompt}
        ]
    )
    content = chat_response.choices[0].message.content
    return content


@retry_decorator
def james(category: str, clue: str) -> str:
    '''PlayerAgent for GPT-3.5-Turbo. Calls OpenAI.'''
    
    prompt = jeopardy_question_template.format(**locals())
    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=system_messages + [
            {"role": "user", "content": prompt}
        ]
    )
    content = chat_response.choices[0].message.content
    return content


@retry_decorator
def david(category: str, clue: str) -> str:
    '''Player agent for llama2:7b. Runs locally.'''
    import ollama
    
    prompt = jeopardy_question_template.format(**locals())
    response = ollama.chat(
        model='llama2',
        messages=system_messages + [
            {"role": "user", "content": prompt}
        ]
    )
    content = response['message']['content']
    return content


@retry_decorator
def larissa(category: str, clue: str) -> str:
    '''PlayerAgent for llama3:8b. Runs locally.'''
    import ollama
    
    prompt = jeopardy_question_template.format(**locals())
    response = ollama.chat(
        model='llama3:8b',
        messages=system_messages + [
            {"role": "user", "content": prompt}
        ]
    )
    content = response['message']['content']
    return content


# TODO
bedrock_client = boto3.client('bedrock-runtime')

@retry_decorator
def brad(category: str, clue: str) -> str:
    '''PlayerAgent for llama3:70b. Uses AWS Bedrock.'''
    
    prompt = jeopardy_question_template.format(**locals())
    messages = system_messages + [ {"role": "user", "content": prompt} ]
    llama3_prompt = format_llama3_prompt(messages)

    input_data = {
        'prompt': llama3_prompt
    }
    
    response = bedrock_client.invoke_model(
        modelId='meta.llama3-70b-instruct-v1:0',
        body=json.dumps(input_data),
    )
    response_data = json.loads(response['body'].read())
    return response_data['generation'].strip()


@retry_decorator
def amy(category: str, clue: str) -> str:
    '''PlayerAgent for a fine-tuned GPT-3.5-Turbo. Calls OpenAI.'''
    
    prompt = jeopardy_question_template.format(**locals())
    chat_response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal:jeopardy1k:9MJuormU",
        messages=system_messages + [
            {"role": "user", "content": prompt}
        ]
    )
    content = chat_response.choices[0].message.content
    return content


@retry_decorator
def alex(category: str, clue: str, correct_response: str, contestant_response: str) -> str:
    '''HostAgent used to judge respondants. This is necessary because
    a correct response may not necessarily match the given answer 100%
    of the time. This uses GPT-4 (calling OpenAI's API) because it makes
    very few judging errors; only about 1%. Alex will always start his
    response with "Correct!" if the contestant got it right, but may include
    additional commentary.
    '''
    
    prompt = f'''
    CATEGORY: {category}
    ORIGINAL CLUE: {clue}
    CORRECT RESPONSE: {correct_response}

    CONTESTANT RESPONSE: {contestant_response}
    '''
    chat_response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.0,
        messages=[
            {"role": "system", "content": ws("""
				You are Alex Trebek, the host of Jeopardy. Given a correct
				answer from the producer and a response (in the form of a
				question) from a contestant, use your best judgement to decide
				if the contestant is correct.  If the contestant answered the
				question correctly and in the form of a question, the first
				word of your response MUST be "Correct!".
            """)},
            {"role": "user", "content": prompt}
        ]
    )
    content = chat_response.choices[0].message.content
    return content


# Oh, what a tangled web we weave, when at first we don't
# design our system to support cross-validation and have to hack
# it in after the fact.
from functools import lru_cache
@lru_cache(maxsize=3)
def old_jeopardy(split_date='2011-01-01'):
    '''Slices the vector database so that only questions from
    before the split date are included. You should then pose
    questions from after the split date so exact matches never occur.
    '''
    vdb = load_jeopardy_embedding_data()
    jeopardy_old = [ q for q in vdb.data if q['air_date'] < split_date ]
    
    split_vector = np.array([q['air_date'] >= split_date for q in vdb.data])
    embeddings_old = vdb.embeddings[~split_vector].copy()
    
    return embeddings_old, jeopardy_old


@retry_decorator
def mattea(category: str, clue: str) -> str:
    '''PlayerAgent implementing the RAG pattern. It only knows the
    exact answers to questions from before 2011-01-01. It uses llama3:8b,
    a fast but fairly dumb model, so we can see the bump from RAG clearly.
    '''
    import ollama
    
    # find similar past questions 
    rag_query = f'CATEGORY: {category}\n{clue}'
    embeddings_old, jeopardy_old = old_jeopardy()
    augment_data = brute_force_search(embeddings_old, jeopardy_old, query=rag_query, k=10)
    augment_json = json.dumps(augment_data)

    # augment the prompt with retrieved questions
    prompt = jeopardy_question_template.format(**locals())

    # this looks cheesy, but other attempts to embed the RAG context
    # resulted in Llama3 being confused about which question's were historical
    # and which was actually being asked. This prompt fixed that.
    messages = system_messages + [
        {"role": "user", "content": f"Correct! By the way, here are three historical questions that may help you answer future questions: {augment_json}"},
        {"role": "assistant", "content": "Thank you, I'll be sure to refer back to those if they help with a question in the future!"},
        {"role": "user", "content": prompt}
    ]

    # use a small, fast model for generation
    response = ollama.chat(
        model='llama3:8b',
        messages=messages
    )
    content = response['message']['content']
    return content



@retry_decorator
def openai_embeddings_create(batch):
    '''OpenAI only allows 2048 chunks max on a single API call.
    Therefore we need to break our database into batches and call the
    API for each one. Since we want to implement retry logic for each
    API call, this operation is split into this separate function.
    Call `create_embeddings_database()` for a higher level interface.
    '''
    return client.embeddings.create(
      model="text-embedding-ada-002",
      input=batch,
      encoding_format="float"
    )


def create_embeddings_database(chunks: List[str], batch_size=2048) -> np.ndarray:
    '''Calls the OpenAI embeddings API (using the ada-002 model) repeatedly
    to encode every chunk string as a 1536 dimensional float32 vector. The
    results will be returned as a 2D matrix with 1536 columns and one row per chunk.
    '''
    database = None
    n = len(chunks)
    
    for batch_index in range(0, n, batch_size):
        # call the OpenAI embeddings API for each batch
        batch = chunks[batch_index:batch_index+batch_size]
        embedding_response = openai_embeddings_create(batch)
        embeddings = embedding_response.data

        # defer creation of the database until we know the embedding dimension
        if database is None:
            m = len(embeddings[0].embedding)    
            database = np.zeros(shape=(n, m), dtype='float32')

        # populate database
        for i, embedding in enumerate(embeddings):
            index = batch_index + i
            database[index, :] = embedding.embedding

    return database


def embed(chunk: str) -> np.ndarray:
    '''Calculates the vector embedding for a single string.
    '''
    db = create_embeddings_database([chunk])
    return db[0]


def best_k(database: np.ndarray, query: np.ndarray, k: int = 5):
    '''Brute force best-k algorithm. Simply computes *all*
    the cosine distances between the query and every embedding
    vector in the database and returns the top-k in sorted order
    (lowest distance first.) Call `brute_force_search()` for
    a higher level interface.
    '''
    # Normalize the query vector
    query = query / np.linalg.norm(query)
    
    # Compute cosine distances
    distances = 1.0 - (database @ query)

    # Find the indices of the k smallest distances
    best_k_unsorted = np.argpartition(distances, k)[:k]

    # Sort these indices by distance
    sorted_indices = np.argsort(distances[best_k_unsorted])
    best_k_sorted = best_k_unsorted[sorted_indices]
    best_k_distances = distances[best_k_sorted]
    
    return best_k_distances, best_k_sorted


def brute_force_search(
    vector_database: np.ndarray,
    jeopardy_data: List[dict],
    query: Union[str, np.ndarray], k=5) -> List[dict]:
    '''Returns `k` jeopardy questions that are the closest
    semantic match to the query. Uses an exhaustive, brute
    force search. The query can be either
    a string (which will be automatically encoded) or an
    embedding vector (if you want to avoid calling the
    embedding API repeatedly.)
    '''        
    # accept either a single string or a single vector
    if isinstance(query, str):
        query = embed(query)
    assert isinstance(query, np.ndarray)
    assert query.shape == (1536,)

    # use the exhaustive cosine similarity search.
    
    distances, indices = best_k(vector_database, query, k=k)
    
    # format the results, including distance to query
    results = [ jeopardy_data[i].copy() for i in indices ]
    for distance, result in zip(distances, results):
        #result['chunk'] = jeopardy_chunk_template.format(**result)
        result['distance'] = distance * 2
    return results


def index_search(
    jeopardy_index, #: faiss.swigfaiss.Index,
    jeopardy_data: List[dict],
    query: Union[str, np.ndarray], k=5) -> List[dict]:
    '''Returns `k` jeopardy questions that are the closest
    semantic match to the query. Uses a faiss HNSW index 
    under the hood to speed things up. The query can be either
    a string (which will be automatically encoded) or an
    embedding vector (if you want to avoid calling the
    embedding API repeatedly.)
    '''        
    # accept either a single string or a single vector
    if isinstance(query, str):
        query = embed(query)
    assert isinstance(query, np.ndarray)
    assert query.shape == (1536,)

    # use the FAISS index to find the k-nearest-neighbors of the query vector
    distance_matrix, index_matrix = jeopardy_index.search(query.reshape(1, -1), k=k)
    indices = index_matrix[0]
    distances = distance_matrix[0]
    
    # format the results, including distance to query
    results = [ jeopardy_data[i].copy() for i in indices ]
    for distance, result in zip(distances, results):
        #result['chunk'] = jeopardy_chunk_template.format(**result)
        result['distance'] = distance
    return results


currency_regex = re.compile(r"[\s,$]+")
def clean_currency(value: Any) -> float:
    '''Removing whitespaces, commas, and dollar signs from a string
    and converts it a float.'''
    if isinstance(value, str):
        value = currency_regex.sub("", value)

    # None, empty string, or zero int/float
    if not value:
        return 0

    # NaN
    if isinstance(value, float) and math.isnan(value):
        return 0
        
    return float(value)


def load_jeopardy_dataset(remove_unfair: bool = False) -> List[dict]:
    '''Loads the raw jeopardy dataset from JSON as one big
    list of dicts, where each dict is a question with category,
    question, answer, etc.
    '''
    filename = os.path.join(JEOPARDY_DATA_DIR, 'jeopardy.json')
    with open(filename) as fin:
        jeopardy_data = json.load(fin)

    for q in jeopardy_data:
        q['value'] = clean_currency(q['value'])

    if remove_unfair:
        jeopardy_data = [ q for q in jeopardy_data if 'href=' not in q['question'] ]

    return jeopardy_data


def load_jeopardy_index(): # -> faiss.swigfaiss.Index:
    '''Loads a serialized faiss index off disk.
    '''
    filename = os.path.join(JEOPARDY_DATA_DIR, 'jeopardy_faiss_hnsw.index')
    return faiss.read_index(filename)


@dataclass
class JeopardyData:
    embeddings: np.ndarray
    chunks: List[str]
    data: List[dict]


def load_jeopardy_embedding_data() -> JeopardyData:
    '''Loads a complete vector database with the vector embeddings,
    chunks used for encoding, and the original question data/metadata.
    '''
    filename = os.path.join(JEOPARDY_DATA_DIR, 'jeopardy_vdb2.joblib')
    data = joblib.load(filename)

    return JeopardyData(
        embeddings= data['vdb'],
        chunks=data['chunks'],
        data=data['questions'])


def generate_hnsw_index(embeddings, filename):
    '''Generate a faiss HNSW index and save it to disk.
    '''
    hnsw_index = faiss.IndexHNSWFlat(1536, 32)
    hnsw_index.add(embeddings)
    faiss.write_index(hnsw_index, filename)


def format_for_fine_tuning(category: str, question: str, answer: str, **kwargs) -> dict:
    '''formats one question/answer pair as one line in an OpenAI fine-tuning .jsonl file.'''
    messages = [
        {"role": "system", "content": "You are a contestant on Jeopardy. Answer in the form of a question."},
        {"role": "user", "content": f'\nCATEGORY: {category}\n{question}'},
        {"role": "assistant", "content": f"What is {answer}?"}
    ]
    return { "messages": messages }


def generate_fine_tuning_data(
    jeopardy_data: List[dict],
    sample_size: int,
    seed: bool = None,
    label: str = 'fine_tuning',
    verbose: bool = False):
    '''Generates a fine-tuning data file in the OpenAI .jsonl format for a given sample size.
    '''
    fine_tuning_filename = os.path.join(JEOPARDY_DATA_DIR, f'jeopardy_{label}_sample_{sample_size}.jsonl')

    # generate the sample in a psuedo-random, repeatable way
    with TemporarySeed(seed or sample_size):
        jeopardy_sample = random.sample(jeopardy_data, sample_size)

    # write each each example as JSON object on it's own line.
    # The full file itself is not valid JSON, but each line is.
    with open(fine_tuning_filename, 'w') as fout:
            for question in jeopardy_sample:
                line = json.dumps(format_for_fine_tuning(**question))
                fout.write(line)
                fout.write('\n')
    
    if verbose:
        print(f'wrote {sample_size} lines to {fine_tuning_filename}')


def jeopardy_dialogue(
    question_data: dict,
    contestant: PlayerAgent,
    host: HostAgent = alex,
    verbose: bool = True) -> Tuple[str, str]:
    '''Handles one question/answer/judgement interaction between the host
    and a contestant. The question is converted to a category and clue
    and passed to the contestant, who answers. Then the original question,
    the correct answer, and the contestant's given answer are passed to
    the host for judgement. 
    '''
    q = question_data
    if verbose:
        print("Category:", q['category'])
    question = q['question'].strip("'")
    if verbose:
        print("Clue:", question)
        print("Answer:", q['answer'])
    contestant_answer = contestant(q['category'], question)
    if verbose:
        print("Contestant:", contestant_answer)
    judgement = alex(q['category'], question, q['answer'], contestant_answer)
    if verbose:
        print("Alex:", judgement)

    return contestant_answer, judgement


def jeopardy_benchmark(contestant, dataset, sample_size=3, verbose=False) -> pd.DataFrame:
    '''collects benchmark data for one contestant by choosing `n` random questions
    from the dataset, putting the question to the contestant agent, and using the `alex()`
    agent to determine correctness.
    '''
    jeopardy_sample = random.sample(dataset, sample_size)
    
    for question_data in jeopardy_sample:
        contestant_answer, judgement = jeopardy_dialogue(question_data, contestant=contestant, verbose=verbose)
        question_data['contestant_answer'] = contestant_answer
        question_data['judgement'] = judgement
        question_data['correct'] = judgement.lower().startswith('correct')
        
        # clean up irrelevant keys
        for key in ['air_date', 'round', 'show_number']:
            if key in question_data:
                del question_data[key]

        # display progress
        if verbose:
            print()
        else:
            print('.', end='', flush=True)
    
    jeopardy_df = pd.DataFrame.from_records(jeopardy_sample)

    return jeopardy_df


def jeopardy_benchmark_suite(
    jeopardy_data: List,
    contestants: List = None,
    sample_size: int = 3,
    seed: int = None, # most random number
    verbose: bool = False) -> pd.DataFrame:
    '''Runs the Jeopardy! benchmark for a number of contestants. All
    contestants receive the exact same set of questions. Results
    are returned in a single dataframe with contestants distinguished 
    by the "label" column.
    '''    
    all_benchmark_results = []
    if contestants is None:
        contestants = [amy, ken, larissa, david, brad, james, mattea]

    if seed is None:
        seed = sample_size

    with TemporarySeed(seed):
        for contestant in contestants:
            if verbose:
                print(f'\nCONTESTANT: {contestant.__name__}\n')
            else:
                print(f'\n{contestant.__name__}', end='', flush=True)

            benchmark_results_df = jeopardy_benchmark(
                contestant,
                dataset=jeopardy_data,
                sample_size=sample_size,
                verbose=verbose)
            benchmark_results_df.insert(0, 'label', contestant.__name__)
            all_benchmark_results.append(benchmark_results_df)

    all_benchmark_results_df = pd.concat(all_benchmark_results)
    return all_benchmark_results_df


def evaluate_jeopardy_benchmark(benchmark_results: pd.DataFrame, label=None, verbose=True) -> dict:
    '''Evaluates the performance of a single contestant, i.e. computes success rate and standard error.
    '''
    successes = benchmark_results['correct'].sum()
    failures = (~benchmark_results['correct']).sum()
    sample_size = successes + failures
    
    # Compute proportion and standard error
    success_rate = successes / sample_size
    safe_success_rate = (successes + 0.5) / (sample_size + 1)
    se = math.sqrt(safe_success_rate * (1 - safe_success_rate) / sample_size)

    # human readable error bars
    margin_of_error = 1.96 * se
    if verbose:
        result = f"{label}'s results: {successes}/{sample_size} = {success_rate:0.3f} ± {margin_of_error:.2f}"
        print(result)
    
    return { 
        "label": label,
        "successes": successes,
        "failures": failures,
        "sample_size": sample_size,
        "success_rate": success_rate,
        "standard_error": se,
    }


def evaluate_jeopardy_benchmarks(benchmark_results: pd.DataFrame, verbose=False) -> pd.DataFrame:
    '''Given a dataframe of results for multiple contestants, create a dataframe with one
    row per contestant showing their benchmark evaluation metrics.
    '''
    # evaluate each contestant's results
    evaluations = []
    for contestant_name in benchmark_results['label'].unique():
        contestant_results_df = benchmark_results[ benchmark_results['label'] == contestant_name ]
        evaluation = evaluate_jeopardy_benchmark(contestant_results_df, label=contestant_name, verbose=False)
        evaluations.append(evaluation)

    # prepare the return dataframe
    jeopardy_benchmark_evaluations_df = pd.DataFrame.from_records(evaluations)
    jeopardy_benchmark_evaluations_df['name'] = jeopardy_benchmark_evaluations_df['label'].apply(full_name)
    jeopardy_benchmark_evaluations_df.sort_values('success_rate', ascending=False, inplace=True)
    
    return jeopardy_benchmark_evaluations_df


def plot_evaluations(jeopardy_benchmark_evaluations_df: pd.DataFrame):
    '''Plots the evaluation dataframe as a barchart with error bars.
    '''
    import matplotlib.pyplot as plt
    
    # Plotting the bar chart with error bars
    dpi = 96
    scale = (892/759)
    plt.figure(figsize=(scale*892/dpi, scale*594/dpi), dpi=dpi)

    # Iterate over the DataFrame to assign colors based on 'name'
    colors = []
    for name in jeopardy_benchmark_evaluations_df['name']:
        if 'gpt' in name.lower():  # Check if 'gpt' is in the name
            colors.append('lightseagreen')  # Shade of green
        elif 'llama' in name.lower():  # Check if 'llama' is in the name
            colors.append('cornflowerblue')  # Shade of blue
        else:
            colors.append('gray')  # Default color
    
    # main bar chart
    plt.bar(
        jeopardy_benchmark_evaluations_df['name'],
        jeopardy_benchmark_evaluations_df['success_rate'],
        yerr=jeopardy_benchmark_evaluations_df['standard_error'] * 1.96, # 95% CI
        #color='lightblue',
        color=colors,
        ecolor='darkblue', # error bar color  
        width=0.4,
        capsize=10)
    
    # what I love about matplotlib, is how intuitive it is
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.set_ticks(np.linspace(0, 1, 11))
    ax.yaxis.grid(color='gray', which='both', alpha=0.25)
    ax.tick_params(axis='x', labelsize=9)
    
    # Adding labels and title
    plt.ylim(0, 1)
    plt.ylabel('Success Rate')
    plt.title('Jeopardy Benchmark Comparisons')
