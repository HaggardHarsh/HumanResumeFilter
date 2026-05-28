# HumanResumeFilter

An end-to-end Python pipeline and **Web Application** that uses **semantic search**, **hybrid BM25 retrieval**, and **LLM re-ranking** to intelligently match and rank candidates against a Job Description.

![Web UI Demo](https://img.shields.io/badge/UI-Modern_Glassmorphism-8b5cf6?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-Flask_|_FAISS_|_SentenceTransformers-3b82f6?style=for-the-badge)

## Architecture

The system can be used via the **beautiful web interface** or as a headless CLI. 

```text
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│  JD + CSV   │───▶│ data_loader  │───▶│   embedder   │───▶│  retriever   │───▶│  llm_scorer   │
│  (Input)    │    │  (Step 1)    │    │  (Step 2)    │    │  (Step 3)    │    │  (Step 4)     │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └───────────────┘
                                                                                       │
                                    ┌──────────────────────┐                           ▼
                                    │ Web UI (app.py) OR   │◀────────────────── ranked_results 
                                    │ CLI (main.py)        │                       (Step 5)
                                    └──────────────────────┘
```

## Key Features
1. **Semantic Search (`FAISS`):** Converts the job description and candidate profiles into dense vectors using `all-MiniLM-L6-v2` to understand the *meaning* of the skills, not just exact words.
2. **Keyword Match (`BM25`):** Fuses the semantic score with an Okapi BM25 keyword search to ensure exact critical terms (e.g., "Kubernetes") aren't missed.
3. **LLM Deep Dive:** Takes the top mathematical matches and prompts an LLM (OpenAI, Gemini, xAI, or Groq) to holistically evaluate the candidate's career trajectory, impact, and cultural fit, returning a 0-100 score and a justification.
4. **Modern Web UI:** A premium, glassmorphic Single Page Application (SPA) powered by Flask to drag-and-drop resumes without touching the terminal.

---

## 🚀 Quick Start (Web App)

The easiest way to use the ranker is through the built-in web dashboard.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```

### 3. Open the UI
From the UI, you can:
- Drag and drop your Job Description (`.txt`) and Candidate Database (`.csv`).
- Adjust the retrieval settings (Semantic Weight, Top N).
- Select your LLM Provider and paste your API Key directly in the UI!

> **Note:** If you don't provide an API key, the system gracefully falls back to using the pure mathematical (Semantic + BM25) retrieval scores.

---

## 💻 CLI Usage

If you prefer to automate the pipeline or run it headlessly, you can use the CLI:

```bash
# Basic run with sample data
python main.py --jd sample_data/sample_jd.txt --candidates sample_data/sample_candidates.csv

# Customise retrieval and LLM settings
python main.py \
  --jd sample_data/sample_jd.txt \
  --candidates sample_data/sample_candidates.csv \
  --retrieval-top-n 30 \
  --llm-top-n 10 \
  --alpha 0.8 \
  --llm-provider xai \
  --llm-model grok-3-mini-fast
```

For the CLI, API keys must be set via environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
# Or: GEMINI_API_KEY, XAI_API_KEY, GROQ_API_KEY
```

## 📄 Candidate CSV Format

Your candidate CSV (`.csv`) should include these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `candidate_id` | No (auto-generated) | Unique identifier |
| `name` | Yes | Full name |
| `title` | Yes | Current job title |
| `skills` | Yes | Comma-separated skills |
| `experience_years` | Yes | Years of experience |
| `education` | No | Degree and institution |
| `career_history` | No | Previous roles summary |
| `certifications` | No | Professional certifications |
| `platform_activity` | No | GitHub, StackOverflow, etc. |
| `bio` | No | Summary / bio text |

## License
MIT
