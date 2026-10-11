# Rag Pipeline Project

A small question-answering project for local text and PDF files. It splits documents into chunks, stores their embeddings in Chroma, and sends the closest chunks to an OpenAI model with the question.

## Setup

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running the project. The example key is a placeholder.

## Run

```bash
python main.py
```

The demo loads `sample_docs/test_document.txt` and asks two questions: one whose answer is in the file and one outside its content. The prompt asks the model to say when the documents do not contain an answer.

## How it works

1. Text and PDF loaders read the selected files.
2. A splitter makes 500-character chunks with 100-character overlap.
3. `text-embedding-3-small` creates embeddings, which are stored in `./chroma_db`.
4. Each question retrieves three chunks and sends their text and source names to `gpt-4o-mini`.
5. The result contains the question, reply, and a list of retrieved file/chunk references.

To use another file, change `sample_doc_path` in `main.py`. To call the pipeline from Python, use `load_and_chunk_documents()`, `build_vector_store()`, and `generate_rag_response()`.

The source list identifies retrieved chunks; it does not check which chunks actually support each sentence of the reply. The refusal is a prompt instruction, not a separate answer check.

## Files

See [FILES.md](FILES.md) for each file's role.
