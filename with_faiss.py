import os
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as FISS
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

# Load pages from a PDF
# loader = PyPDFLoader("data/National Budget Speech FY2022-23 English Version.pdf")
# pages = loader.load_and_split()

# Load pages from a website
loader = WebBaseLoader(WEBSITE_URLS)
pages = loader.load_and_split()

faiss_obj_path = "models/buffer_salaries.pickle"

if os.path.exists(faiss_obj_path):
    # Load the FAISS object from disk
    faiss_index = FAISS.load(faiss_obj_path)
else:
    # Build and save the FAISS object
    faiss_index = FAISS.from_documents(pages, embeddings, index_name="buffer_salaries")
    faiss_index.save(faiss_obj_path)

messages = [
    SystemMessage(
        content="You are a helpful assistant that know everything about buffer.com. You can answer any question about "
                "the salaries of the employees."),
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
