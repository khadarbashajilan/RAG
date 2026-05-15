from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Process PDF
loader = PyPDFLoader("Marcus-Aurelius-Meditations.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(loader.load())

# 2. Setup Embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 3. Push to Pinecone
# (Assumes index 'meditations-mistral' already exists with 1024 dims)
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="meditations-mistral"
)

print(f"Pushed {len(chunks)} chunks.")
