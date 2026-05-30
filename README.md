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
* Conversation memory for multi-turn interactions
* Vector similarity search using Pinecone
* Grounded responses based on retrieved passages
* CLI-based chat interface
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
CLI Interface
     │
     ▼
Conversation Memory
     │
     ▼
Retriever (Pinecone)
     │
     ▼
Relevant Passages
     │
     ▼
Prompt Construction
     │
     ▼
Mistral LLM
     │
     ▼
Grounded Response
```

---

## Technical Stack

### Backend

* Python
* LangChain
* Mistral AI

### Vector Database

* Pinecone

### Interface

* Rich (CLI)

### AI Concepts Implemented

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Text Chunking
* Embeddings
* Vector Similarity Retrieval
* Conversational Memory
* Prompt Engineering

---

## How It Works

### 1. Document Processing

The PDF version of *Meditations* is loaded and divided into manageable text chunks suitable for embedding and retrieval.

### 2. Embedding Generation

Each chunk is converted into vector embeddings using Mistral AI's embedding models.

### 3. Vector Storage

Generated embeddings are stored in Pinecone for efficient similarity search.

### 4. Retrieval

When a user submits a query, the system retrieves the most relevant passages from the vector database.

### 5. Context Augmentation

Retrieved passages are combined with conversation history to create an enriched prompt.

### 6. Response Generation

The language model generates a response grounded in both the retrieved content and ongoing conversation context.

---

## Project Structure

```text
.
├── Marcus-Aurelius-Meditations.pdf
├── db.py
├── main.py
├── ui_cli.py
├── pyproject.toml
└── .env.example
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
* Mistral AI API Key

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
PINECONE_API_KEY=your_pinecone_api_key
```

### Build the Vector Database

Process *Meditations*, generate embeddings, and populate the Pinecone vector index:

```bash
uv run db.py
```

### Run the Application

Start the interactive Stoic assistant:

```bash
uv run main.py
```

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
* Large Language Model Integration
* Retrieval-Augmented Generation (RAG)
* LangChain
* Vector Databases
* Prompt Engineering
* Semantic Search
* AI Application Development
* System Design
* Conversational AI

---

## License

MIT License

---

*"You have power over your mind — not outside events. Realize this, and you will find strength."*
— Marcus Aurelius
