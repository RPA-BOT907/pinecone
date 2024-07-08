import os
import streamlit as st
from langchain_community.llms import Ollama 
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone

# Get Pinecone API key from system environment variables
key = os.getenv("Pinecone_Api_key")
if key is None:
    raise ValueError("Pinecone API key not found in environment variables")

# Connect to Pinecone DB
pc = Pinecone(api_key=key)
index = pc.Index("langchain")

# Query Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = PineconeVectorStore(index, embeddings)

# Define a function to get matching results from Pinecone
def matching_results(query, k=2):
    return vectorstore.similarity_search(query, k=k)

# Initialize the LLM model
llm = Ollama(model="mistral")

# Create the QA chain
chain = load_qa_chain(llm=llm, chain_type="stuff")

# Define the function to get the model's answer
def model_answer(user_input):
    doc_search = matching_results(user_input)
    response = chain.run(input_documents=doc_search, question=user_input)
    return response

# Streamlit UI
st.title("PDF Q&A Chat BOT")

quotation = st.text_input("Enter your Query:")

if quotation:
  # Get the model's answer
  answer = model_answer(quotation)
  response_data = answer
  # Display the answer on Streamlit UI
  st.write(response_data)