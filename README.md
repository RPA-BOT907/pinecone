Building a PDF Q&A Chatbot with LangChain, Pinecone, and Streamlit
Introduction
Creating an intelligent chatbot that can answer questions from a large repository of documents is a powerful application of modern natural language processing (NLP) and machine learning. In this blog, we will build a Q&A chatbot using LangChain, Pinecone, and Streamlit. The chatbot will leverage a language model to process user queries and return relevant information from a document database.

Prerequisites
Before diving into the code, ensure you have the following prerequisites:

Python installed on your system.
Pinecone API key.
Streamlit for building the web interface.
LangChain for managing the language model and chain operations.
Pinecone for vector database services.
Step-by-Step Guide
1. Setup and Import Libraries
First, we need to import the necessary libraries and modules:

python
Copy code
import os
import streamlit as st
from langchain_community.llms import Ollama 
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone
2. Retrieve Pinecone API Key
Fetch the Pinecone API key from your system environment variables. If the key is not found, raise an error:

python
Copy code
key = os.getenv("Pinecone_Api_key")
if key is None:
    raise ValueError("Pinecone API key not found in environment variables")
3. Connect to Pinecone Database
Initialize the connection to Pinecone and specify the index to be used:

python
Copy code
pc = Pinecone(api_key=key)
index = pc.Index("langchain")
4. Query Embeddings
Set up the embeddings using the OllamaEmbeddings model:

python
Copy code
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = PineconeVectorStore(index, embeddings)
5. Define Function for Matching Results
Create a function that searches for similar documents based on the query:

python
Copy code
def matching_results(query, k=2):
    return vectorstore.similarity_search(query, k=k)
6. Initialize the LLM Model
Instantiate the language model using Ollama:

python
Copy code
llm = Ollama(model="mistral")
7. Create the QA Chain
Set up the question-answering chain using the initialized language model:

python
Copy code
chain = load_qa_chain(llm=llm, chain_type="stuff")
8. Define Function to Get Model's Answer
Write a function that uses the model to generate answers based on user input:

python
Copy code
def model_answer(user_input):
    doc_search = matching_results(user_input)
    response = chain.run(input_documents=doc_search, question=user_input)
    return response
9. Build Streamlit Interface
Create a simple user interface using Streamlit where users can enter their queries and receive answers:

python
Copy code
st.title("PDF Q&A Chat BOT")

quotation = st.text_input("Enter your Query:")

if quotation:
    answer = model_answer(quotation)
    st.write(answer)
Conclusion
With this setup, you now have a fully functional Q&A chatbot that can answer questions based on a large repository of documents. This implementation can be extended and customized for various use cases, such as customer support, educational purposes, or document analysis.