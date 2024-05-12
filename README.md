JPT is short for JeopardyGPT and is a small library for applying LLM models
such as ChatGPT to Jeopardy! trivia data. It was written as an adjunct to
[this blog post][OLJ] and is not otherwise useful.

[OLJ]: https://www.oranlooney.com/post/jeopardy/


Most of the interesting code is in `jpt.py`. Several Jupyter notebooks are
included to demonstrate how to use it. `util.py` includes a handful of generic
utilities that get copied from project to project.  `server.py` is a very rough
FastAPI server that exposes the functionality of `jpt` as a JSON REST endpoint.
`bedrock_streaming.py` is demo of streaming tokens out of the AWS Bedrock
service.


Setup
-----

Unfortunately, because this benchmark compares models across several providers,
installing the prerequisites is a minor nightmare.

Download the [Kaggle Jeopardy! dataset][JKD].

In `jpt.py`, set `JEOPARDY_DATA_DIR` to directory containing `jeopardy.json`.

Install [ollama][OLL] and run:

    ollama pull llama2:7b
    ollama pull llama3:8b

Sign up for an AWS developer account if you haven't already.  In your user home
directory, create an `.aws` directory containing your AWS credentials in the
standard way.

Sign up for an OpenAI developer account if you haven't already.  In your user
home directory, create an `.openai` directory containing a `credentials.yaml`
file in this format:

    organization: "*******"
    api_key: "********"

Run `pip -r requirements.txt`.

Open the `Jeopardy Fine Tuning` Jupyter notebook and generate the fine tuning
files. Use the 1k or 10k version to create a fine tuning job on OpenAI. Put
the resulting model ID in the `amy()` function in jpt.py.

Open the `Jeopardy Vector Database` Jupyter notebook and generate the embeddings
file. This will take about 30 minutes. Optionally create the faiss HNSW index;
this takes only about 1 minute.

You can now open the `Jeopardy Benchmarks` Jupyter notebook and run the benchmarks.
This takes between 1 and 2 hours at a sample size of 1000.

Note that the main entry point, `jpt.jeopardy_benchmark_suite()`, allows you to 
specify a list of contestants to run, so if you don't install everything above,
you can still run it for the ones you have.


[JKD]: https://www.kaggle.com/datasets/aravindram11/jeopardy-dataset-updated
[OLL]: https://ollama.com/
