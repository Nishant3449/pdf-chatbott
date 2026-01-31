from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
logging.basicConfig(level=logging.DEBUG)


DATA_PATH ="data/" 
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
## print("no of pages: ", len(documents))

# create chunks
def create_chunks(ectracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=50)
    text_chunks=text_splitter.split_documents(ectracted_data)
    return text_chunks

chunks = create_chunks(documents)
## print("Chunks created: ", len(chunks))

# create vector embedding

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# store embedding in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)