# Marcus Aurelius Meditations RAG

A Retrieval-Augmented Generation (RAG) application that delivers context-aware Stoic guidance grounded in the teachings of Marcus Aurelius' *Meditations*. The system combines semantic search, conversation memory, and large language models to generate responses based on relevant source passages rather than relying solely on model knowledge.

---

## Overview

Large language models are powerful conversational tools, but they often lack grounding in specific source material. This project addresses that limitation by implementing a RAG pipeline that retrieves relevant passages from *Meditations* and uses them as context for response generation.

The result is an AI assistant capable of providing practical Stoic insights while remaining anchored to the philosophical principles found in the original text.

---

## Key Features

* Retrieval-Augmented Generation (RAG) architecture
* Semantic search over *Meditations*
* Context-aware conversational responses
* Conversation memory via SQLite (wiped on each `./run.sh` launch)
* Automatic conversation summarization to manage context length
* Vector similarity search using Pinecone
* Grounded responses based on retrieved passages
* CLI-based chat interface with live spinner and Markdown-rendered panels
* Automatic fallback across 11 free Gemini models on daily-quota exhaustion
* Manual model switching and mid-session memory reset via slash commands
* Modular and extensible codebase

---

## Example Interaction

### User

> I'm feeling anxious about a presentation tomorrow.

### Assistant

> Marcus Aurelius frequently reminded himself that external events are not fully within our control, but our judgments about them are. Focus your energy on preparation, clarity, and effort rather than the outcome. The presentation itself is temporary; your character and response to it are what truly matter.

---

## System Architecture

```text
User Query
     │
     ▼
CLI Interface (Rich)
     │
     ▼
LangGraph Agent  ──  Summarization Middleware (over 2000 tokens, keep last 10)
     │
     ├─► SQLite Checkpointer (wiped on ./run.sh launch)
     └─► Search Tool
            │
            ▼
      Heuristic Query Cleanup (local, no API call)
            │  "how do i dweal w fwar?" → "how do i deal with fear?"
            ▼
      Mistral Embeddings (mistral-embed, 1024-dim)
            │
            ▼
      Pinecone MMR (meditations-mistral, k=3)
            │
            ▼
      Relevant Passages
            │
            ▼
       Gemini (active model in the free rotation, 4096 tokens)
            │
            ▼
      Grounded Response
            │
            ▼
      Daily quota hit?  →  rotate to next free model, rebuild agent
```


---

## Technical Stack

### Backend

* Python 3.12+
* LangChain + LangGraph (agent framework)
* Google Gemini free-tier rotation (default `gemini-3.5-flash` plus 10 fallback flash models, `max_output_tokens=4096`, `thinking_level=low` on Gemini 3+ models only)
* Mistral Embed (embeddings, 1024 dims)
* Query preprocessing via heuristic cleanup (typo/filler stripping, out-of-scope detection) — no API call
* Automatic model rotation when one model's free-tier daily quota runs out, with backoff retry for per-minute throttling
* `/model`, `/models`, `/clear`, `/help` commands to inspect and switch the rotation and reset memory

### Vector Database

* Pinecone (serverless)

### Persistence

* SQLite (conversation checkpoints via langgraph-checkpoint-sqlite)

### Interface

* Rich (CLI)

### AI Concepts Implemented

* Retrieval-Augmented Generation (RAG)
* Semantic Search with MMR (Maximal Marginal Relevance)
* Text Chunking
* Embeddings
* Vector Similarity Retrieval
* Conversational Memory (SQLite, wiped on each `./run.sh` launch)
* Conversation Summarization (automatic context management)
* Prompt Engineering
* Tool-use Agents

---

## How It Works

### 0. Query Preprocessing

User queries are messy — typos, all-caps rants, conversational rambling. When the agent calls the `search_meditations` tool, a local heuristic pass (no API call, no tokens spent) normalizes the query to lowercase, fixes known typos, strips filler words, and detects out-of-scope queries (e.g. "best pizza in town"). An out-of-scope query short-circuits with a canned "outside the scope of Stoic philosophy" reply and never touches Pinecone.

### 1. Document Processing

The PDF version of *Meditations* is loaded and divided into manageable text chunks suitable for embedding and retrieval.

### 2. Embedding Generation

Each chunk is converted into 1024-dimensional vector embeddings using Mistral's mistral-embed model.

### 3. Vector Storage

Generated embeddings are stored in a Pinecone serverless index (`meditations-mistral`) for efficient similarity search.

### 4. Retrieval

When a user submits a query, the system retrieves the 3 most relevant passages using Maximal Marginal Relevance (MMR) to balance relevance and diversity.

### 5. Agent Processing

A LangGraph agent receives the query, searches the vector store via a tool, and generates a response grounded in the retrieved passages. The agent maintains conversation history in a SQLite database and automatically summarizes older messages to manage context length.

### 6. Response Generation

The language model generates a response grounded in both the retrieved content and ongoing conversation context, delivered via a Rich CLI interface. If the response is truncated due to the token limit, a visible warning is shown below the panel.

### 7. Quota Management

Google enforces free-tier daily quotas per model, so exhausting one model does not have to end the session. A per-minute `429` triggers exponential backoff (using the `retryDelay` the API returns, capped at 60s, up to 4 attempts). A *daily* quota error marks that model as spent, moves to the next model in the rotation, and rebuilds the agent around it. When every model is spent, the CLI prints the Pacific-time reset.

---

## CLI Commands

| Command | Effect |
|---|---|
| `/exit` (or `exit`, `0`) | Quit |
| `/clear` | Delete the conversation checkpoint for the current thread — clean slate without restarting |
| `/model` | Switch to another Gemini model by number or name; the agent is rebuilt |
| `/models` | List the rotation, marking models whose daily quota is spent |
| `/help` | Show the command list |

---

## Project Structure

```text
.
├── Marcus-Aurelius-Meditations.pdf
├── db.py                      # Vector DB ingest script
├── main.py                    # Agent + checkpointer + tool + query cleanup
├── models.py                  # Free-tier Gemini model rotation
├── ui_cli.py                  # Rich CLI chat loop
├── pyproject.toml
├── .env.example
├── .gitignore
├── checkpoints.db             # SQLite conversation memory (deleted by run.sh)
└── run.sh                     # Env validation, memory wipe, launch
```

---

## Engineering Challenges

### Maintaining Philosophical Consistency

Ensuring responses remain aligned with Stoic principles while still addressing modern-day questions.

### Retrieval Quality

Selecting chunk sizes and retrieval strategies that maximize relevance without losing context.

### Context Management

Balancing retrieved passages and conversation memory within model context limits.

### Hallucination Reduction

Grounding responses in retrieved source material to improve factual and thematic consistency.

---

## Getting Started

### Prerequisites

* Python 3.12+
* Pinecone Account
* Google AI Studio API Key (free tier available)
* Mistral API Key (free tier available)

### Installation

Clone the repository:

```bash
git clone https://github.com/khadarbashajilan/RAG.git
cd RAG
```

Install dependencies using uv:

```bash
uv sync
```

### Environment Configuration

Create a `.env` file from the provided example:

```bash
cp .env.example .env
```

Update the values inside `.env` with your credentials:

```env
MISTRAL_API_KEY=your_mistral_api_key
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Optional — override the model rotation and LLM tuning (comma-separated rotation, tried in order):

```env
GEMINI_MODELS=gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash
THINKING_LEVEL=low
MAX_OUTPUT_TOKENS=4096
TEMPERATURE=0.7
```

`THINKING_LEVEL` is only sent to Gemini 3+ models (2.5 rejects it) and `TEMPERATURE` is only sent to non-`lite` models.

Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
Get your free Mistral API key from [Mistral AI Console](https://console.mistral.ai).

> **Do not also export `GOOGLE_API_KEY`.** The `google-genai` SDK prefers `GOOGLE_API_KEY` over `GEMINI_API_KEY` and warns when both are set. `load_dotenv()` does not override variables that already exist in the environment, so a shell export silently overrides your `.env` value and the app runs on a different key than you think. If you see `Both GOOGLE_API_KEY and GEMINI_API_KEY are set`, remove the shell export.

### Build the Vector Database

Process *Meditations*, generate embeddings, and populate the Pinecone vector index:

```bash
uv run db.py
```

Chunk IDs are derived from a SHA1 of the page content, so re-running is safe — already-stored chunks are skipped.

### Run the Application

```bash
./run.sh
```

`run.sh` verifies the three required keys, creates the virtualenv if missing, deletes `checkpoints.db` for a clean slate, then launches the CLI.

Running `uv run main.py` directly also works, but skips env validation and keeps any existing conversation history.

## Future Improvements

* Web interface using Streamlit or React
* Source citations for retrieved passages
* Hybrid search (keyword + vector retrieval)
* Evaluation metrics for retrieval accuracy
* Support for additional Stoic philosophers
* Docker deployment
* API endpoint for external integrations

---

## Why I Built This

I've always been fascinated by how timeless philosophical ideas can remain relevant in modern life. While studying Stoicism, I realized that many people struggle to apply philosophical concepts to real-world situations despite their practical value.

This project allowed me to explore Retrieval-Augmented Generation while building something personally meaningful: an AI system that transforms a classic philosophical text into an interactive, conversational experience.

Beyond the technical implementation, the project reflects my interest in creating AI systems that are grounded, useful, and capable of making complex knowledge more accessible.

---

## Skills Demonstrated

* Python Development
* Large Language Model Integration (Google Gemini)
* Retrieval-Augmented Generation (RAG)
* LangChain + LangGraph Agent Framework
* Vector Databases (Pinecone)
* Persistent Conversation Memory (SQLite)
* Prompt Engineering
* Semantic Search with MMR
* AI Application Development
* System Design
* Conversational AI
* Error Handling & Rate Limit Management
* Free-Tier Quota Management with Multi-Model Fallback

---

## License

MIT License

---

*"You have power over your mind — not outside events. Realize this, and you will find strength."*
— Marcus Aurelius
