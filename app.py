import os
import streamlit as st
from langchain_community.llms import ollama 
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone

# Get Pinecone API key from system environment variables
key = os.getenv("Pinecone_Api_key")
if key is None:
    st.error("Pinecone API key not found in environment variables")
    st.stop()

# Connect to Pinecone DB.
pc = Pinecone(api_key=key)
index = pc.Index("langchain")

# Query Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = PineconeVectorStore(index, embeddings)

# Define a function to get matching results from Pinecone
def matching_results(query, k=2):
    return vectorstore.similarity_search(query, k=k)

# Initialize the LLM model
llm = ollama(model="mistral")

# Create the QA chain
chain = load_qa_chain(llm=llm, chain_type="stuff")

# Define the function to get the model's answer
def model_answer(user_input):
    try:
        doc_search = matching_results(user_input)
        response = chain.run(input_documents=doc_search, question=user_input)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon=":robot_face:")

st.title("PDF Q&A Chatbot")
st.subheader("Ask any question related to the documents and get instant answers!")

with st.form(key="query_form"):
    quotation = st.text_area("Enter your Query:", height=150)
    submit_button = st.form_submit_button(label="Submit")

if submit_button and quotation:
    with st.spinner("Processing your query..."):
        answer = model_answer(quotation)
    st.write("### Answer")
    st.write(answer)

st.markdown("""
    <style>
    .stTextArea, .stButton>button {
        width: 100%;
        font-size: 18px;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)
