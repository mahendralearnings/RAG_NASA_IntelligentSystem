"""
RAGAS Evaluator — Scores RAG response quality.

Three metrics:
- Faithfulness    : Is answer grounded in context?
- ResponseRelevancy: Does answer address the question?  
- RougeScore      : Word overlap between answer and context

Uses Claude as judge LLM + free local embeddings.
NO OpenAI key needed.
"""

import os
import json
import statistics
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
load_dotenv()

# LangChain wrappers that RAGAS needs
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# Claude via LangChain
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_LANGCHAIN_AVAILABLE = True
except ImportError:
    ANTHROPIC_LANGCHAIN_AVAILABLE = False

# Free local embeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_AVAILABLE = False
    
def _build_evaluator_components():
    """
    Create the LLM + embeddings RAGAS needs to score answers.
    
    WHY Claude as judge?
    RAGAS needs an LLM to evaluate faithfulness and relevancy.
    Claude Haiku is cheap + accurate enough for evaluation.
    
    WHY local embeddings?
    ResponseRelevancy needs embeddings to compare
    question vs answer similarity. Free local model works fine.
    
    Returns: (evaluator_llm, evaluator_embeddings, error_string)
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return None, None, "ANTHROPIC_API_KEY not set in .env"

    if not ANTHROPIC_LANGCHAIN_AVAILABLE:
        return None, None, "Run: pip install langchain-anthropic"

    if not HUGGINGFACE_AVAILABLE:
        return None, None, "Run: pip install langchain-community"

    try:
        # Claude Haiku as judge — temperature=0 for consistent scoring
        evaluator_llm = LangchainLLMWrapper(
            ChatAnthropic(
                model="claude-haiku-4-5",
                anthropic_api_key=anthropic_key,
                temperature=0
            )
        )

        # Free local embeddings for relevancy scoring
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        return evaluator_llm, evaluator_embeddings, ""

    except Exception as e:
        return None, None, str(e)
    


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    openai_key: Optional[str] = None  # ignored, kept for compatibility
) -> Dict[str, float]:
    """
    Score ONE question-answer pair using RAGAS metrics.
    Called after every chat response in real-time.
    
    Args:
        question : user's question
        answer   : Claude's generated answer  
        contexts : list of raw chunks retrieved from ChromaDB
    
    Returns:
        {"faithfulness": 0.87, "answer_relevancy": 0.92, "rouge_score": 0.74}
        or {"error": "message"} if something fails
    """
    # Input validation — handle empty/malformed inputs gracefully
    if not question or not question.strip():
        return {"error": "Empty question provided"}
    if not answer or not answer.strip():
        return {"error": "Empty answer provided"}
    if not contexts or not any(c.strip() for c in contexts):
        return {"error": "No valid context provided"}

    if not RAGAS_AVAILABLE:
        return {"error": "Run: pip install ragas datasets"}

    # Build evaluator components
    evaluator_llm, evaluator_embeddings, err = _build_evaluator_components()
    if err:
        return {"error": err}

    try:
        # Define metrics with our Claude judge
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            ),
        ]

        # RAGAS dataset — only needs these 3 columns now
        data = {
            "user_input":         [question],
            "response":           [answer],
            "retrieved_contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        result = evaluate(dataset=dataset, metrics=metrics)

        # Extract scores from result DataFrame
        scores = {}
        skip_cols = {"user_input", "response", "retrieved_contexts"}
        result_row = result.to_pandas().iloc[0].to_dict()

        for key, value in result_row.items():
            if key not in skip_cols:
                try:
                    scores[key] = round(float(value), 3)
                except (TypeError, ValueError):
                    pass

        return scores if scores else {"error": "No scores returned"}

    except Exception as e:
        return {"error": f"RAGAS failed: {str(e)[:150]}"}