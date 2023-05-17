import os
import requests
import mimetypes
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
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

load_dotenv()

# Get the Variables from the .env file
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

DEFAULT_MILVUS_CONNECTION = {
    "host": os.getenv(""),
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}


class MilvusManager:
    def __init__(self, host, port):
        self.client = Milvus(host=host, port=port)

    def list_collections(self):
        return self.client.list_collections()

    def create_collection(self, collection_name, fields):
        self.client.create_collection(collection_name, fields)

    def delete_collection(self, collection_name):
        self.client.drop_collection(collection_name)


class MilvusCollectionManager:
    def __init__(self, milvus_manager, collection_name):
        self.milvus_manager = milvus_manager
        self.collection_name = collection_name

    def collection_exists(self):
        active_collections = self.milvus_manager.list_collections()
        return self.collection_name in active_collections

    def create_collection(self, fields):
        self.milvus_manager.create_collection(self.collection_name, fields)

    def delete_collection(self):
        self.milvus_manager.delete_collection(self.collection_name)


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


class DocumentLoaderFactory:
    @staticmethod
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


def train_or_load_model(train, file_path, name_space):
    if train:
        loader = DocumentLoaderFactory.get_loader(file_path)
        pages = loader.load_and_split()

        print("Updating the model")
        milvus_text_vector_store = MilvusTextVectorStore.from_texts(
            texts=pages,
            embedding=embeddings,
            collection_name=MILVUS_COLLECTION_NAME,
            drop_old=True  # This will drop the existing collection and create a new one
        )
    else:
        print("Loading the model")
        milvus_text_vector_store = MilvusTextVectorStore.from_existing_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            embedding=embeddings
        )

    return milvus_text_vector_store


def answer_questions(milvus_text_vector_store):
    messages = [
        SystemMessage(
            content='I want you to act as a document that I am having a conversation with. Your name is "AI '
                    'Assistant". You will provide me with answers from the given info. If the answer is not included, '
                    'say exactly "Hmm, I am not sure." and stop after that. Refuse to answer any question not about '
                    'the info. Never break character.')
    ]
    while True:
        question = input("Ask a question (type 'stop' to end): ")
        if question.lower() == "stop":
            break

        docs = milvus_text_vector_store.search_texts(query=question, k=1)

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
    file_path = "data/cuet.pdf"
    name_space = "cuet"

    train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
    milvus_text_vector_store = train_or_load_model(train, file_path, name_space)
    answer_questions(milvus_text_vector_store)


if __name__ == "__main__":
    main()
