import os
from dotenv import load_dotenv

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


csv_loader = CSVLoader(file_path='data/property.csv')

index_creator = VectorstoreIndexCreator()
document_search = index_creator.from_loaders([csv_loader])

chat_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff",
                                         retriever=document_search.vectorstore.as_retriever(), input_key="question")
query_text = "Claim Reference ID is KRN6/2/2/E/30/0/0/7 what is Gazette Published Date for it"
response = chat_chain({"question": query_text})

print(response)
print(response['result'])
