# langchain-chat

langchain-chat is a powerful AI-driven Q&A system that leverages OpenAI's GPT-4 model to provide relevant and accurate
answers to user queries. The system indexes documents from websites or PDF files using FAISS (Facebook AI Similarity
Search) and offers a convenient interface for interacting with the data.

## Features

- Load and split documents from websites or PDF files
- Index documents using FAISS for efficient similarity search
- Utilize OpenAI's GPT-4 to generate human-like responses
- Remember previous conversations and provide context-aware answers
- Easy to set up and extend

## Installation

1. Clone the repository
2. Create a virtual environment
    ```bash
    python -m venv venv
    ```
3. Activate the virtual environment
    ```bash
    source venv/Scripts/activate
    ```
4. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```
5. Copy the `.env.example` file to `.env` and fill in the required values
   ```bash
   cp .env.example .env
   ```
   ```bash
   OPEN_AI_KEY = "sk-"
   WEBSITE_URLS="https://website1, https://website2"
   ```
6. Run the application
    ```bash
    python with_faiss.py
    ```

## Example Images

![Example 1](https://github.com/shamspias/langchain-chat/blob/main/images/conversation.PNG)