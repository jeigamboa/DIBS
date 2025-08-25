### Preliminaries
import os
from dotenv import load_dotenv

# Import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# Import supabase
from supabase.client import Client, create_client

### Main

# Load environment variables
load_dotenv()  

# Initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load pdf docs from folder 'documents'
file_path = "C:/David/000 Work Prep and Independent Proj/" # Replace with path of DIBS on your system

pdf_loader = PyPDFDirectoryLoader(file_path + "DIBS/RAG Ingest/documents")
pdf_docs = pdf_loader.load()

# Load csv docs from folder 'csvs'
csv_docs = []
csv_folder = file_path + "DIBS/RAG Ingest/csvs"
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        csv_docs.extend(loader.load())

# Combine documents with csvs
all_docs = pdf_docs + csv_docs

# Split the documents in multiple chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

# Store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=800,
)

