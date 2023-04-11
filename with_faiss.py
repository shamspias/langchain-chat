import os
import pickle
import faiss
import numpy as np
import requests
import mimetypes
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as FISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader, UnstructuredURLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit

from dotenv import load_dotenv

load_dotenv()


class FAISS(FISS):
    """
    FAISS is a vector store that uses the FAISS library to store and search vectors.
    """

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


# Get the Variables from the .env file
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
WEBSITE_URL = os.getenv('WEBSITE_URLS')
WEBSITE_URLS = WEBSITE_URL.split(",")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)


def get_loader(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type == 'application/pdf':
        return PyPDFLoader(file_path)
    elif mime_type == 'text/csv':
        return CSVLoader(file_path)
    elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


# Load pages from a website

def is_valid_url(url):
    """
    Check if a URL is valid
    """
    parsed_url = urlsplit(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)


def extract_links(url):
    """
    Extract all links from a URL
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            absolute_url = urljoin(url, href)
            if is_valid_url(absolute_url):
                links.append(absolute_url)

    return links


def extract_links_from_websites(websites):
    """
    Extract all links from a list of websites
    """
    all_links = []

    for website in websites:
        links = extract_links(website)
        all_links.extend(links)

    return all_links


train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
if train:
    faiss_obj_path = "models/langchain.pickle"
    loader = get_loader("data/3.pdf")
    pages = loader.load_and_split()

    if os.path.exists(faiss_obj_path):
        # Load the FAISS object from disk
        faiss_index = FAISS.load(faiss_obj_path)
        # make new embeddings
        new_embeddings = faiss_index.from_documents(pages, embeddings, index_name="langchain", dimension=1536)

        # Add the new embeddings to the FAISS object
        # new_faiss_index = merge_vectors(faiss_index.index, new_embeddings.index)
        # faiss_index.index = new_faiss_index

        # Save the FAISS object
        new_embeddings.save(faiss_obj_path)

    else:
        # Build and save the FAISS object

        faiss_index = FAISS.from_documents(pages, embeddings, index_name="langchain", dimension=1536)
        faiss_index.save(faiss_obj_path)

else:
    faiss_obj_path = "models/langchain.pickle"
    faiss_index = FAISS.load(faiss_obj_path)

messages = [
    SystemMessage(
        content="You know everything about langchain and you can give any solution and write code for any problem.")
]

while True:
    question = input("Ask a question (type 'stop' to end): ")
    if question.lower() == "stop":
        break

    docs = faiss_index.similarity_search(query=question, k=2)

    main_content = question + "\n\n"
    for doc in docs:
        main_content += doc.page_content + "\n\n"

    messages.append(HumanMessage(content=main_content))
    ai_response = chat(messages).content
    messages.pop()
    messages.append(HumanMessage(content=question))
    messages.append(AIMessage(content=ai_response))

    print(ai_response)
