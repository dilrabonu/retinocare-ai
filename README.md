
# RetinoCare AI

**An end-to-end diabetic retinopathy screening agent** combining a PyTorch classification pipeline with a retrieval-augmented (RAG) recommendation system — built as a production-oriented portfolio project demonstrating applied ML engineering and agentic AI system design.

> ⚠️ **Not a diagnostic tool.** RetinoCare AI is a screening/triage support system. It never issues a diagnosis and always directs the user to consult a qualified ophthalmologist. This principle is enforced in code, not just documentation 
---

## What this project demonstrates

- **Applied deep learning**: trained and rigorously compared three CNN architectures on real medical imaging data, selecting a production model based on more than raw accuracy
- **Agentic AI system design**: a hybrid-retrieval RAG pipeline grounding LLM output in real source documents, with fail-closed safety guarantees
- **Production engineering discipline**: config-driven training, CI/CD, containerization, a tested API, and a documented model-selection rationale

---

## Live demo

![Diabetic Retinopathy Screening Assistant](docs/image.png)

Given a fundus image, the system returns a severity classification, a confidence score, and a guideline-grounded recommendation with citations:

> **Severity: Severe** (confidence: 51.1%)
>
> *"Urgent referral is warranted when screening identifies Severe or Proliferative diabetic retinopathy... this indicates urgent (same-week) ophthalmologist evaluation would be the appropriate pathway."* — cited from `referral-criteria.md`
>
> *This is a screening support tool, not a medical diagnosis. Please consult a qualified ophthalmologist for evaluation and care decisions.*

Note how the agent explicitly flags when its confidence is only moderate ("should be treated as provisional input for clinical judgment, not a settled result") — this is deliberate: the agent's language is grounded in the model's actual confidence score, not generic reassurance.

---

## Architecture

Fundus image
│
▼
PyTorch classifier (ResNet18, fine-tuned)  ──►  severity (0–4) + confidence score
│
▼
Hybrid RAG retrieval (ChromaDB dense + BM25 sparse, fused via Reciprocal Rank Fusion)
│
▼
Claude API  ──►  grounded, cited recommendation
│
▼
Fail-closed disclaimer enforcement  ──►  final response (FastAPI / Streamlit)

Full pipeline: `src/retinocare/` — classification (`models/`), retrieval + agent (`agents/`), API (`api/`).

---

## Model comparison — engineering decision, not just a leaderboard

Three architectures were trained and evaluated identically on a held-out test set from [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection):

| Model            | Weighted F1 | Macro F1 | Severe F1 | Proliferative F1 | ECE (calibration) |
|------------------|-------------|----------|-----------|-------------------|--------------------|
| Baseline CNN (from scratch) | 0.6907 | 0.4780 | 0.2553 | 0.1587 | 0.0688 |
| ResNet18 (transfer learning) | 0.7952 | 0.6265 | 0.3385 | 0.5333 | **0.0547** |
| EfficientNet-B0 (transfer learning) | **0.8133** | **0.6550** | **0.4063** | **0.5517** | 0.0966 |

**EfficientNet-B0 has the highest raw F1. ResNet18 was chosen for production anyway.**

Why: the downstream RAG agent phrases its recommendation based on the model's confidence score (see the "moderate confidence" language in the demo above). Expected Calibration Error (ECE) measures whether a stated confidence can actually be trusted — ResNet18's confidence scores are meaningfully more reliable (ECE 0.055 vs 0.097). A screening tool that says "90% confident" and is wrong more than 10% of the time is a worse foundation than a slightly less accurate model whose confidence numbers mean what they say.

This trade-off — and the reasoning behind it — is documented in full in [`docs/model_comparison_results.md`](docs/model_comparison_results.md), along with individual training writeups for each model in `docs/results_*.md`.

---

## Safety design

This isn't a bolt-on disclaimer — it's enforced in code and covered by tests:

- **Fail-closed disclaimer**: `RAGAgent.respond()` checks whether the LLM's output includes the exact required disclaimer text. If it's missing for any reason, the code appends it before returning — the safety guarantee never depends solely on the LLM remembering. Verified by `tests/test_agent.py::test_agent_response_always_includes_disclaimer`.
- **Grounded, cited responses**: the system prompt explicitly forbids fabricating clinical information beyond what was retrieved from `knowledge_base/guidelines/`, and every recommendation cites its source document.
- **Explicit uncertainty language**: the agent is prompted to reflect the model's actual confidence level rather than presenting every prediction with uniform authority.

---

## Tech stack

| Layer | Technology |
|---|---|
| Modeling | PyTorch, torchvision, albumentations |
| Serving | FastAPI, Streamlit |
| Retrieval | ChromaDB (dense) + BM25 (sparse), Reciprocal Rank Fusion |
| Agent / LLM | Anthropic Claude API |
| Testing | pytest (10 tests: dataset integrity, agent safety, API behavior) |
| Infra | Docker, GitHub Actions CI |

---

## Repository structure

```
retinocare-ai/
├── data/                    # Dataset (unzipped)
├── models/                  # Trained models (.pth)
├── knowledge_base/          # Guidelines for RAG
│   └── guidelines/
├── src/retinocare/
│   ├── models/            # Training scripts
│   ├── agents/            # RAG + LLM integration
│   ├── api/               # FastAPI server
│   └── streamlit_app.py   # Web UI
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── ...
```
---

## Setup

```bash
git clone https://github.com/dilrabonu/retinocare-ai.git
cd retinocare-ai

conda create -n retinocare-ai python=3.11 -y
conda activate retinocare-ai
pip install -r requirements.txt

cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

**Get the dataset and train:**
```bash
# See notebooks/01_eda.ipynb for the full download + preprocessing walkthrough
python -m src.retinocare.models.train --config configs/train_config.yaml --model resnet18
```

**Run the full application** (two terminals):
```bash
uvicorn src.retinocare.api.main:app --reload      # backend, port 8000
streamlit run streamlit_app/app.py                 # frontend, port 8501
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Results summary

- Trained and compared 3 model architectures with a rigorous, config-driven pipeline
- Achieved 0.80 weighted F1 (ResNet18, production model) on a 5-class severity task with real clinical imbalance
- Built a hybrid dense+sparse RAG retrieval system with measurable, tested safety guarantees
- Delivered a working end-to-end demo: image upload → classification → grounded recommendation → mandatory disclaimer, in under 2 seconds

---

## Roadmap

- [x] **Phase 1**: Classification model comparison + RAG agent (this README)
- [ ] **Phase 2**: Voice agent layer (LiveKit STT/TTS)
- [ ] **Phase 3**: Multi-agent orchestration (LangGraph: triage → referral → scheduling)

---

## Author

**Dilrabo Khidirova** — Senior Lecturer (Machine Learning), IT Park University, Fergana, Uzbekistan. ML Engineer candidate with a background in production ML pipelines, computer vision, and applied NLP.

## License

MIT — see [`LICENSE`](LICENSE)