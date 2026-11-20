
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing!")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("Supabase connected successfully!")


# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
