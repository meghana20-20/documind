# DocuMind 🧠📄
### RAG-based Document Q&A Assistant with Hallucination Detection

DocuMind lets you drop in a folder of documents (PDF, TXT, DOCX) and ask
natural-language questions about them. Unlike most tutorial-style RAG demos,
DocuMind doesn't just return an answer — it tells you **how confident it is
that the answer is actually grounded in your documents**, and shows you the
exact source chunks used, so you can catch hallucinations before they bite you.

## Why this project is different

Most "RAG chatbot" student projects stop at: embed → retrieve → generate.
That's the easy 80%. DocuMind adds the harder, more interesting 20%:

- **Hallucination / groundedness scoring** — after the LLM generates an
  answer, DocuMind re-embeds each sentence of the answer and checks its
  semantic similarity against the retrieved source chunks. Low-similarity
  sentences are flagged in the UI as "not well supported by sources."
- **Source-attributed citations** — every answer is returned with the exact
  chunk(s) and document(s) it was built from, not just a generic "Source: doc1.pdf".
- **Pluggable LLM backend** — works with OpenAI, Anthropic, or a fully local
  Ollama model, so you can demo it without paying for API credits.
- **Chunk-level transparency UI** — a Streamlit sidebar view lets you inspect
  what was actually retrieved for any query, which is great for explaining
  your project in interviews.

## Architecture

```
 ┌─────────────┐     ┌──────────────┐     ┌────────────────┐
 │  Documents   │────▶│   Chunking   │────▶│  Embeddings     │
 │ (pdf/docx/txt)│    │ (ingest.py)  │     │ (sentence-      │
 └─────────────┘     └──────────────┘     │  transformers)  │
                                            └────────┬────────┘
                                                      ▼
                                            ┌──────────────────┐
                                            │  FAISS Vector    │
                                            │  Store            │
                                            └────────┬─────────┘
                                                      ▼
 Question ──────────────────────────────▶  Retrieve Top-K Chunks
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │  LLM (OpenAI /   │
                                            │  Anthropic /     │
                                            │  Ollama)         │
                                            └────────┬─────────┘
                                                      ▼
                                            ┌──────────────────┐
                                            │ Hallucination     │
                                            │ / Groundedness    │
                                            │ Scorer             │
                                            └────────┬─────────┘
                                                      ▼
                                            Answer + Citations
                                            + Confidence Score
```

## Tech Stack

- Python 3.10+
- `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
- `faiss-cpu` for vector similarity search
- `streamlit` for the UI
- `pypdf` / `python-docx` for document parsing
- LLM: OpenAI API, Anthropic API, or local Ollama (`llama3`, `mistral`, etc.)

## Setup

```bash
git clone <your-repo-url>
cd documind
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your LLM backend in `.env` (copy from `.env.example`):

```
LLM_PROVIDER=ollama        # options: openai, anthropic, ollama
OPENAI_API_KEY=            # if using openai
ANTHROPIC_API_KEY=         # if using anthropic
OLLAMA_MODEL=llama3        # if using ollama (must be running locally)
```

## Run

```bash
streamlit run app.py
```

Then open the local URL, upload documents from `data/sample_docs/` (or your
own), and start asking questions.

## Project Structure

```
documind/
├── app.py                  # Streamlit UI
├── src/
│   ├── ingest.py            # Load + chunk documents
│   ├── embed_store.py       # FAISS vector store wrapper
│   ├── qa_engine.py         # RAG pipeline + groundedness scoring
│   └── llm_client.py        # Pluggable LLM backend
├── data/sample_docs/        # Sample documents to try it on
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Sample Demo Flow (for your resume/interview talking points)

1. Upload 2-3 PDFs (e.g., lecture notes, a research paper).
2. Ask a question that's clearly answered in the docs → get a high
   confidence score (green) with citations.
3. Ask a question that's *not* covered in the docs → DocuMind flags the
   answer as poorly grounded (red) instead of confidently making something up.

This is the core "interview story": *"I built a RAG system that doesn't just
answer questions — it tells you when it doesn't actually know."*

## Possible Extensions

- Swap FAISS for a hosted vector DB (Pinecone/Weaviate) for scale.
- Add multi-document cross-referencing ("compare doc A and doc B on X").
- Fine-tune the groundedness threshold using a labeled eval set.
- Add conversation memory for multi-turn Q&A.

## License

MIT
