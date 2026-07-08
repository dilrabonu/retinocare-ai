"""RAG agent: takes a classification result, retrieves relevant guideline
excerpts, and calls the Claude API to produce a grounded, cited recommendation.
"""

import os

from anthropic import Anthropic

from src.retinocare.agents.prompts import MANDATORY_DISCLAIMER, SYSTEM_PROMPT, build_agent_prompt
from src.retinocare.agents.retriever import HybridRetriever

SEVERITY_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


class RAGAgent:
    def __init__(
        self,
        knowledge_base_dir: str,
        model: str = "claude-sonnet-5",
        retriever: HybridRetriever | None = None,
    ):
        # retriever can be injected (useful for testing without ChromaDB/network)
        self.retriever = retriever or HybridRetriever(knowledge_base_dir)
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def respond(self, severity: str, confidence: float, k: int = 4) -> dict:
        """Returns {"text":..., "sources":[...], "disclaimer_included": bool}.

        Fail-closed: if the model's response omits the exact disclaimer text,
        this method appends it before returning -- the disclaimer must never
        depend solely on the LLM remembering to include it.
        """
        retrieved = self.retriever.retrieve(query=f"screening guidance for {severity} diabetic retinopathy", k=k)

        prompt = build_agent_prompt(severity, confidence, retrieved)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(block.text for block in response.content if block.type == "text")

        disclaimer_included = MANDATORY_DISCLAIMER in text
        if not disclaimer_included:
            text = f"{text}\n\n{MANDATORY_DISCLAIMER}"

        sources = sorted({chunk["source"] for chunk in retrieved})

        return {
            "text": text,
            "sources": sources,
            "disclaimer_included": True,  # true by construction after the fail-closed append above
        }