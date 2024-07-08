#from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Pinecone
from PyPDF2 import PdfReader
import time
from langchain_core.documents import Document
import os
# Load PDF file
pdf_path = "C:\\Users\\haris\\OneDrive\\Desktop\\Child-Budget_removed.pdf"

# Initialize the PDF reader
pdf_file = PdfReader(pdf_path)

# Read text from PDF
raw_text = ''
for i, page in enumerate(pdf_file.pages):
    content = page.extract_text()
    if content:
        raw_text += content

#text splitting
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 200,
    length_function = len,
)

text = text_splitter.split_text(raw_text)

docs = [Document(page_content=texts) for texts in text]

# Text embedding using OllamaEmbeddings
ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

de=ollama_emb.embed_query("Sample text")

print(ollama_emb)
print(len(de))


#get key from env
key = os.getenv("Pinecone_Api_key")



#pinecone DB connaction
pinecone_api_key=key
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=pinecone_api_key)




index_name = "langchain"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        index = pc.Index(index_name)
        print(index)
        time.sleep(1)
    print("Index created")
else:
    print("Index already exists")


""""
 #store vectors in DB
from langchain_pinecone import PineconeVectorStore
index_names="langchain"
vectorstore = PineconeVectorStore.from_documents(
       docs,
       ollama_emb,
       index_name=index_names      
        
)



print("hello")
print(vectorstore)
"""