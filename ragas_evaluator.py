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
    
def load_test_questions(filepath: str) -> List[Dict]:
    """Load test questions from JSON or txt file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test file not found: {filepath}")

    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        questions = []
        for item in data:
            if isinstance(item, str):
                questions.append({"question": item, "category": "general"})
            elif isinstance(item, dict) and "question" in item:
                questions.append(item)
        return questions

    # .txt format
    questions = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[1].strip():
                    questions.append({
                        "question": parts[1].strip(),
                        "category": "general"
                    })
    return questions


def run_batch_evaluation(
    test_filepath: str,
    collection,
    n_retrieve: int = 3,
    output_filepath: str = "batch_eval_results.json"
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on ALL questions in test file.
    Outputs per-question scores + aggregate mean per metric.
    
    This proves your RAG system works systematically —
    not just on one question but across multiple categories.
    """
    import rag_client
    import llm_client

    questions = load_test_questions(test_filepath)
    if not questions:
        return {"error": f"No questions found in {test_filepath}"}

    print(f"\n{'='*50}")
    print(f"BATCH EVALUATION — {len(questions)} questions")
    print(f"{'='*50}\n")

    per_question_results = []
    all_scores: Dict[str, List[float]] = {}

    for i, q_item in enumerate(questions):
        question = q_item.get("question", "").strip()
        category = q_item.get("category", "general")
        if not question:
            continue

        print(f"[{i+1}/{len(questions)}] {category}: {question[:60]}...")

        result_entry = {
            "question_number": i + 1,
            "question":        question,
            "category":        category,
            "answer":          "",
            "scores":          {},
            "error":           None
        }

        try:
            # Retrieve chunks
            docs_result = rag_client.retrieve_documents(
                collection, question, n_retrieve
            )
            if not docs_result or not docs_result.get("documents"):
                result_entry["error"] = "No documents retrieved"
                per_question_results.append(result_entry)
                continue

            # Format context
            raw_docs  = docs_result["documents"][0]
            raw_metas = docs_result["metadatas"][0]
            raw_dists = docs_result.get("distances", [[]])[0]
            context   = rag_client.format_context(raw_docs, raw_metas, raw_dists)

            # Generate answer
            answer = llm_client.generate_response(
                openai_key="",
                user_message=question,
                context=context,
                conversation_history=[]
            )
            result_entry["answer"] = answer

            # Score with RAGAS
            scores = evaluate_response_quality(question, answer, raw_docs)
            result_entry["scores"] = scores

            if "error" not in scores:
                for metric, value in scores.items():
                    if isinstance(value, (int, float)):
                        all_scores.setdefault(metric, []).append(value)

            score_str = ", ".join(
                f"{k}={v:.3f}" for k, v in scores.items()
                if isinstance(v, float)
            )
            print(f"  Scores: {score_str}")

        except Exception as e:
            result_entry["error"] = str(e)
            print(f"  Error: {e}")

        per_question_results.append(result_entry)

    # Aggregate mean per metric
    aggregate_scores = {}
    for metric, values in all_scores.items():
        if values:
            aggregate_scores[metric] = {
                "mean":  round(statistics.mean(values), 3),
                "min":   round(min(values), 3),
                "max":   round(max(values), 3),
                "count": len(values)
            }

    report = {
        "test_file":              test_filepath,
        "total_questions":        len(questions),
        "evaluated_successfully": sum(
            1 for r in per_question_results if not r.get("error")
        ),
        "aggregate_scores":       aggregate_scores,
        "per_question_results":   per_question_results
    }

    with open(output_filepath, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print("AGGREGATE SCORES (mean across all questions):")
    for metric, stats in aggregate_scores.items():
        print(f"  {metric:25s}: mean={stats['mean']:.3f} "
              f"min={stats['min']:.3f} max={stats['max']:.3f}")
    print(f"\nResults saved to: {output_filepath}")

    return report