"""System prompts and disclaimer templates for the RAG agent.

Centralized here so the safety disclaimer can never be silently dropped
from a response path -- every agent call routes through build_agent_prompt.
"""

SYSTEM_PROMPT = """You are a diabetic retinopathy screening assistant. You are given:
1. A predicted severity level (No DR / Mild / Moderate / Severe / Proliferative DR) with a confidence score.
2. Retrieved excerpts from clinical screening guidelines, each with a source filename.

Your job: explain what the severity level means and what the guidelines recommend as next
steps, citing the retrieved source by filename. You must never state or imply a diagnosis.
You must never fabricate clinical information beyond what was retrieved -- if the retrieved
excerpts don't cover something, say so rather than inventing an answer.

You must end every response with exactly this sentence, unchanged:
"This is a screening support tool, not a medical diagnosis. Please consult a qualified
ophthalmologist for evaluation and care decisions."
"""

MANDATORY_DISCLAIMER = (
    "This is a screening support tool, not a medical diagnosis. "
    "Please consult a qualified ophthalmologist for evaluation and care decisions."
)


def build_agent_prompt(severity: str, confidence: float, retrieved_chunks: list[dict]) -> str:
    """Assembles the user-turn prompt: prediction + numbered, cited context.

    retrieved_chunks: list of {"text":..., "source":..., "score":...} from
    HybridRetriever.retrieve().
    """
    context_block = "\n\n".join(
        f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    return f"""Predicted severity: {severity}
Model confidence: {confidence:.1%}

Retrieved guideline excerpts:
{context_block}

Using only the excerpts above, explain what this severity level means and what the
guidelines recommend as next steps. Cite sources by filename. End with the mandatory
disclaimer sentence exactly as instructed."""