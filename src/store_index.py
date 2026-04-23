
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import downlaod_embedding, filter_to_minimal_docs, load_pdf_files, text_splitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

extracted_documents = load_pdf_files(str(DATA_DIR))
minimal_docs = filter_to_minimal_docs(extracted_documents)
text_chunks = text_splitter(minimal_docs)
embedding_model = downlaod_embedding()
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-rag"
existing_indexes = {index_info["name"] for index_info in pinecone.list_indexes()}
if index_name not in existing_indexes:
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )




docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding_model,
    index_name=index_name,
)
