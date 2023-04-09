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
from langchain.text_splitter import TextSplitter

from dotenv import load_dotenv

load_dotenv()

# Get the Variables from the .env file
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

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

# Load pages from a PDF
# loader = PyPDFLoader("data/National Budget Speech FY2022-23 English Version.pdf")
# pages = loader.load_and_split()

# Load pages from a website
loader = WebBaseLoader("https://buffer.com/salaries")
pages = loader.load_and_split()

# Embed the pages and Upload embeddings to FISS
faiss_index = FAISS.from_documents(pages, embeddings, index_name="buffer_salaries")
print(faiss_index)

question = input("Ask a que   stion: ")
docs = faiss_index.similarity_search(query=question, k=2)

main_content = question + "\n\n"
for doc in docs:
    main_content += doc.page_content + "\n\n"

messages = [
    SystemMessage(
        content="You are a helpful assistant that know everything about buffer.com. You can answer any question about "
                "the salaries of the employees."),
    HumanMessage(content=main_content)
]

print(chat(messages).content)

# AIMessage(content="J'aime programmer.", additional_kwargs={})

# # Embed the pages and Upload embeddings to Pinecone
# docsearch = Pinecone.from_documents(pages, embeddings, index_name=index, namespace="buffer_salaries")
#
# # Query Pinecone index
# query = "how much Jenny gets from buffer.com"
# docs = docsearch.similarity_search(query)
#
# # Clean up
# index.delete()
