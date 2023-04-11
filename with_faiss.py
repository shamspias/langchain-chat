import os
import pickle
from abc import ABC, abstractmethod
from typing import List
import requests
import mimetypes
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as FISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
WEBSITE_URL = os.getenv('WEBSITE_URLS')
WEBSITE_URLS = WEBSITE_URL.split(",")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)


class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass


class FAISS(FISS):
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


def train_or_load_model(train, faiss_obj_path, file_path):
    if train:
        loader = get_loader(file_path)
        pages = loader.load_and_split()

        if os.path.exists(faiss_obj_path):
            faiss_index = FAISS.load(faiss_obj_path)
            new_embeddings = faiss_index.from_documents(pages, embeddings, index_name="langchain", dimension=1536)
            new_embeddings.save(faiss_obj_path)
        else:
            faiss_index = FAISS.from_documents(pages, embeddings, index_name="langchain", dimension=1536)
            faiss_index.save(faiss_obj_path)

        return FAISS.load(faiss_obj_path)
    else:
        return FAISS.load(faiss_obj_path)


def answer_questions(faiss_index):
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


def main():
    faiss_obj_path = "models/langchain.pickle"
    file_path = "data/3.pdf"

    train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
    faiss_index = train_or_load_model(train, faiss_obj_path, file_path)
    answer_questions(faiss_index)


if __name__ == "__main__":
    main()
