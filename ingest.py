
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing in .env")

# Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Supabase connected successfully!")


# Step 1: Load the PDF
loader = PyPDFLoader("data/company_policy.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# Step 3: Generate embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print("Embeddings generated successfully!")


# Step 4: Store documents and embeddings in Supabase
batch_size = 50

for i in range(0, len(chunks), batch_size):

    batch_chunks = chunks[i:i + batch_size]
    batch_vectors = vectors[i:i + batch_size]

    records = []

    for chunk, vector in zip(batch_chunks, batch_vectors):

        records.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vector
        })

    supabase.table("documents").insert(records).execute()

    print(f"Uploaded {len(records)} chunks to Supabase.")


print("Document ingestion completed successfully!")
