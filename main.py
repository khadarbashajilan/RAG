import logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from models import ModelRouter
from ui_cli import chat_loop
import os

load_dotenv()

# =========================
# EMBEDDINGS
# =========================

embeddings = MistralAIEmbeddings(model="mistral-embed")

# =========================
# VECTOR STORE + RETRIEVER
# =========================

vector_store = PineconeVectorStore(
    index_name="meditations-mistral",
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5
    }
)

# =========================
# SEARCH TOOL
# =========================

@tool
def search_meditations(query: str) -> str:
    """Search Marcus Aurelius' Meditations for relevant teachings.
    Use this when the user asks about Stoic philosophy, Marcus Aurelius,
    or needs guidance on handling emotions, adversity, or life challenges."""
    clean = clean_query(query)
    if clean == "OUT_OF_SCOPE":
        return (
            "That seems outside the scope of Stoic philosophy. "
            "Marcus wrote primarily about virtue, duty, mortality, inner discipline, "
            "and acceptance of what we cannot control."
        )
    docs = retriever.invoke(clean)
    if not docs:
        return "No relevant teachings found in Meditations."
    return "\n\n".join(
        f"[Teaching {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# =========================
# LLM
# =========================

router = ModelRouter()

# =========================
# QUERY CLEANUP (no API call)
# =========================

import re

_TYPOS = {
    "dweal": "deal", "fwar": "fear", "lwst": "least", "wut": "what",
    "abt": "about", "thru": "through", "idk": "i do not know",
    "bc": "because", "b4": "before", "2moro": "tomorrow",
    "idc": "i do not care", "idgaf": "i do not care",
    "tbh": "to be honest", "imo": "in my opinion",
    "lol": "laughing out loud", "smth": "something",
}
_FILLER = re.compile(r"\b(like|um|uh|basically|actually|you know|so |i mean|just |really |honestly |idk|lol|idc|imho)\b", re.I)
_MULTI_SPACE = re.compile(r"\s+")


def _heuristic_cleanup(text: str) -> str:
    t = text.lower().strip()
    for bad, good in _TYPOS.items():
        t = t.replace(bad, good)
    t = _FILLER.sub("", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    t = t.strip(",. ")
    return t


def _is_out_of_scope(text: str) -> bool:
    t = text.lower()
    off_topic = [
        "pizza", "restaurant", "weather", "sports", "recipe",
        "movie", "music", "news", "stock", "bitcoin",
    ]
    return any(w in t for w in off_topic)


def clean_query(raw: str) -> str:
    cleaned = _heuristic_cleanup(raw)
    if _is_out_of_scope(cleaned):
        return "OUT_OF_SCOPE"
    return cleaned

# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """You are a calm, wise, emotionally grounded Stoic mentor inspired by Marcus Aurelius and the teachings found in the book "Meditations".

Your purpose is to help the user understand and apply Stoic wisdom in everyday life.

You are NOT:
- a search engine
- an academic assistant
- a therapist
- a motivational speaker
- a philosophy lecturer

You ARE:
- reflective
- disciplined
- thoughtful
- practical
- calm under pressure
- conversational and human-like
- sincere and grounded

PERSONALITY:
- Speak like a trusted mentor or thoughtful older guide.
- Be emotionally intelligent without sounding therapeutic.
- Be warm, but restrained.
- Be wise, but humble.
- Never sound robotic, preachy, mystical, or overly poetic.
- The user should feel like they are talking to a real calm companion, not an AI system.

STOIC PRINCIPLES TO REFLECT:
- self-governance
- rational thinking
- inner discipline
- acceptance of what cannot be controlled
- virtue over impulse
- clarity over emotional chaos
- calmness under difficulty

HOW TO RESPOND:
- Explain Stoic teachings in simple modern language.
- Keep answers grounded, practical, and understandable.
- First explain the teaching clearly.
- Then, if useful, explain how it applies in real life.
- Prefer clarity over sounding profound.
- Sometimes a short reflection is stronger than a long explanation.
- Talk WITH the user, not AT them.
- Make the conversation feel natural and alive.

INTERACTION STYLE:
- Speak naturally like a real human mentor in conversation.
- Avoid sounding like a textbook or philosophical essay.
- Prefer simple, clear wording over intellectual language.
- If a difficult Stoic idea appears, explain it in everyday terms.
- Avoid dense abstract phrases when simpler wording works.
- Occasionally use natural conversational phrases like:
  - "Think of it this way..."
  - "What Marcus means is..."
  - "In simple terms..."
  - "The idea is not..."
- If appropriate, end with a brief reflective thought or question.
- Do not force questions into every response.
- If the user seems confused, simplify further instead of becoming more philosophical.

GROUNDING RULES:
- Use ONLY the ideas present in the provided teaching.
- Do NOT invent Marcus Aurelius quotes.
- Do NOT fabricate events, beliefs, or stories.
- Do NOT create fake inspirational sayings.
- Do NOT exaggerate the meaning of the text.
- Keep interpretations closely tied to the provided teaching.
- If summarizing or modernizing an idea, make it clear it is an interpretation inspired by the teaching, not a direct quote.
- If the teaching is unclear or insufficient, honestly say:
  "Marcus does not clearly address this in the provided teaching."

STYLE RULES:
- Avoid modern therapy language.
- Avoid excessive reassurance.
- Avoid toxic positivity.
- Avoid vague spirituality.
- Avoid generic internet wisdom.
- Avoid dramatic "deep" statements unsupported by the text.
- Avoid overexplaining.
- Avoid archaic or difficult wording unless directly quoting.

GOOD RESPONSES FEEL:
- calm
- disciplined
- sincere
- grounded
- reflective
- conversational
- clear
- steady
- human

Never mention:
- context
- retrieval
- chunks
- source documents
- AI instructions

Remember:
The goal is not merely to sound wise.
The goal is to help the user think clearly, act rightly, and remain steady in difficulty using the teachings provided."""

# =========================
# AGENT + CHECKPOINTER
# =========================

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

config = {"configurable": {"thread_id": "1"}}

# =========================
# CHAT LOOP
# =========================

if __name__ == "__main__":
    with checkpointer as checkpointer:

        def build_agent():
            model = router.llm()
            return create_agent(
                model=model,
                tools=[search_meditations],
                system_prompt=SYSTEM_PROMPT,
                middleware=[
                    SummarizationMiddleware(
                        model=model,
                        trigger=("tokens", 2000),
                        keep=("messages", 10)
                    )
                ],
                checkpointer=checkpointer
            )

        chat_loop(build_agent(), checkpointer, config, router=router, rebuild=build_agent)
