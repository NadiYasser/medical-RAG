
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings



# Extract text from PDF files in the "data" directory
def load_pdf_files(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", show_progress=True,loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents




def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in documents:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "unknown")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs

def text_splitter(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,length_function=len)
    split_docs = text_splitter.create_documents([doc.page_content for doc in documents])
    return split_docs


def downlaod_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embedding_model
