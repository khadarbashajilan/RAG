from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import hashlib
import os
import re
import time

load_dotenv()

# 1. Process PDF
loader = PyPDFLoader("Marcus-Aurelius-Meditations.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(loader.load())
print(f"Loaded {len(chunks)} chunks from PDF.")

# 2. Setup Embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 3. Create Pinecone index if not exists
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "meditations-mistral"

if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created.")
else:
    print(f"Index '{index_name}' already exists.")

# 4. Push to Pinecone
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

BATCH_SIZE = 20
MAX_ATTEMPTS = 8


def chunk_id(doc):
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
    return f"p{doc.metadata.get('page', 0)}-{digest[:16]}"


def with_retry(documents, ids):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            vector_store.add_documents(documents=documents, ids=ids)
            return
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                raise
            match = re.search(r"retryDelay': '(\d+(?:\.\d+)?)s", str(e))
            wait = float(match.group(1)) + 2 if match else min(2 ** attempt, 60)
            print(f"  Retry {attempt}/{MAX_ATTEMPTS} in {wait:.0f}s")
            time.sleep(wait)


all_ids = [chunk_id(doc) for doc in chunks]
existing = set(pc.Index(index_name).fetch(ids=all_ids).vectors.keys())
pending = [(i, d) for i, d in zip(all_ids, chunks) if i not in existing]

print(f"{len(existing)} chunks already stored, {len(pending)} to embed.")

for start in range(0, len(pending), BATCH_SIZE):
    batch = pending[start:start + BATCH_SIZE]
    with_retry([d for _, d in batch], [i for i, _ in batch])
    done = min(start + BATCH_SIZE, len(pending))
    print(f"  {done}/{len(pending)} upserted")

total = pc.Index(index_name).describe_index_stats().total_vector_count
print(f"Index '{index_name}' now holds {total} vectors.")
