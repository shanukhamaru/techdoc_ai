from dotenv import load_dotenv
import os



# Load values from .env file
load_dotenv()

print("[LangSmith] Tracing Project:", os.getenv("LANGCHAIN_PROJECT"))
print("[LangSmith] Tracing Enabled:", os.getenv("LANGCHAIN_TRACING_V2"))


# --- LangSmith Tracing ---
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT


os.environ["LANGCHAIN_TRACING_V2"] = "true"


# Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")
