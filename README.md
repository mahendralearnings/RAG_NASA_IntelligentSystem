# 🚀 NASA Mission Intelligence — RAG Chat System

A production-ready RAG (Retrieval-Augmented Generation) system
for querying NASA mission documents using semantic search and Claude AI.

## Missions Covered
- Apollo 11 (First Moon Landing)
- Apollo 13 (Emergency Mission)
- Challenger STS-51L (Disaster Analysis)

## Architecture
```
NASA Documents → Embedding Pipeline → ChromaDB
                                          ↓
User Question → RAG Client (Search) → Claude AI → Answer
                                          ↓
                                   RAGAS Evaluation
                                   (Quality Scores)
```

## Tech Stack
- **Embeddings**: sentence-transformers (FREE, local)
- **Vector DB**: ChromaDB (local persistent)
- **LLM**: Claude Haiku (Anthropic)
- **Evaluation**: RAGAS (Faithfulness + Relevancy)
- **UI**: Streamlit

## Setup Instructions

### 1. Clone and install
```bash
git clone <your-repo-url>
cd NASA-Intel-Starter
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set up API key
```bash
# Create .env file
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env
```

### 3. Build the knowledge base (run once)
```bash
python embedding_pipeline.py \
  --data-path data_text \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --batch-size 1
```

### 4. Launch the chat UI
```bash
streamlit run chat.py
```

Opens at: http://localhost:8501

### 5. Run batch evaluation
```bash
python ragas_evaluator.py \
  --test-file test_questions.json \
  --chroma-dir ./chroma_db
```

## Project Files

| File | Purpose |
|------|---------|
| `embedding_pipeline.py` | Process NASA docs → ChromaDB |
| `rag_client.py` | Semantic search + context formatting |
| `llm_client.py` | Claude API integration |
| `ragas_evaluator.py` | RAGAS quality scoring |
| `chat.py` | Streamlit chat UI |
| `test_questions.json` | Evaluation test set |
| `evaluation_dataset.txt` | Sample Q&A dataset |

## Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.867 |
| Answer Relevancy | varies by question |

## Key Design Decisions

1. **Free embeddings**: Used sentence-transformers instead of OpenAI
   — saves cost, runs locally, no API key needed for indexing

2. **Chunk size 300**: Smaller chunks due to RAM constraints on
   development machine. Production would use 500-1000.

3. **Honest RAG**: Claude refuses to hallucinate — when context
   is insufficient, it says so clearly.

4. **Mission filtering**: Users can restrict search to one mission
   for more focused answers.