import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader

from dotenv import load_dotenv

load_dotenv()

# Get the Variables from the .env file
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
WEBSITE_URL = os.getenv('WEBSITE_URLS')
WEBSITE_URLS = WEBSITE_URL.split(",")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Get a list of Pinecone indexes
active_indexes = pinecone.list_indexes()

# Check if the Pinecone index exists
index_exists = PINECONE_INDEX_NAME in active_indexes

# Create the Pinecone index if it doesn't exist
if not index_exists:
    index = pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine")
else:
    index = pinecone.Index(index_name=PINECONE_INDEX_NAME)
