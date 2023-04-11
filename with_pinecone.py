import os
import pinecone
import pickle
import mimetypes
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as Pinec
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

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
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

# Get a list of Pinecone indexes
active_indexes = pinecone.list_indexes()

# Check if the Pinecone index exists
index_exists = PINECONE_INDEX_NAME in active_indexes

# Create the Pinecone index if it doesn't exist
if not index_exists:
    index = pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine")
else:
    index = pinecone.Index(index_name=PINECONE_INDEX_NAME)


class Pinecone(Pinec):
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


class URLHandler:
    @staticmethod
    def is_valid_url(url):
        parsed_url = urlsplit(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc)

    @staticmethod
    def extract_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                absolute_url = urljoin(url, href)
                if URLHandler.is_valid_url(absolute_url):
                    links.append(absolute_url)

        return links

    @staticmethod
    def extract_links_from_websites(websites):
        all_links = []

        for website in websites:
            links = URLHandler.extract_links(website)
            all_links.extend(links)

        return all_links


def get_loader(file_path_or_url):
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        handle_website = URLHandler()
        return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
    else:
        mime_type, _ = mimetypes.guess_type(file_path_or_url)

        if mime_type == 'application/pdf':
            return PyPDFLoader(file_path_or_url)
        elif mime_type == 'text/csv':
            return CSVLoader(file_path_or_url)
        elif mime_type in ['application/msword',
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return UnstructuredWordDocumentLoader(file_path_or_url)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")


def train_or_load_model(train, pinecone_obj_path, file_path, index_name):
    if train:
        loader = get_loader(file_path)
        pages = loader.load_and_split()

        if os.path.exists(pinecone_obj_path):
            pinecone_index = Pinecone.load(pinecone_obj_path)
            new_embeddings = pinecone_index.from_documents(pages, embeddings, index_name=index_name)
            new_embeddings.save(pinecone_obj_path)
        else:
            pinecone_index = Pinecone.from_documents(pages, embeddings, index_name=index_name)
            pinecone_index.save(pinecone_obj_path)

        return Pinecone.load(pinecone_obj_path)
    else:
        return Pinecone.load(pinecone_obj_path)


def answer_questions(pinecone_index):
    messages = [
        SystemMessage(
            content="You are a good at resume analysis you can analysis any resume and give good output.")
    ]

    while True:
        question = input("Ask a question (type 'stop' to end): ")
        if question.lower() == "stop":
            break

        docs = pinecone_index.similarity_search(query=question, k=2)

        main_content = question + "\n\n"
        for doc in docs:
            main_content += doc.page_content + "\n\n"

        messages.append(HumanMessage(content=main_content))
        ai_response = chat(messages).content
        messages.pop()
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=ai_response))

        print(ai_response)


def main():
    faiss_obj_path = "models/personal.pickle"
    file_path = "data/sam.pdf"
    index_name = index

    train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
    pinecone_index = train_or_load_model(train, faiss_obj_path, file_path, index_name)
    answer_questions(pinecone_index)


if __name__ == "__main__":
    main()
