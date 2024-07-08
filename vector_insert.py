from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document  # Add this import

# Load PDF file
pdf_path = "C:\\Users\\haris\\OneDrive\\Desktop\\Child-Budget_removed.pdf"
pdf_file = PdfReader(pdf_path)

# Read text from PDF
raw_text = ''
for page in pdf_file.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Text splitting
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 200,
    length_function = len,
)
   #Convert text to chunks
chunks = text_splitter.split_text(raw_text)

# Convert chunks to Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Initialize Ollama embeddings
ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

# Check embedding dimension
sample_embedding = ollama_emb.embed_query("Sample text")
dimension = len(sample_embedding)
print(sample_embedding)
print(f"The dimension of the embeddings is: {dimension}")

# Pinecone setup
pinecone_api_key = "937db33d-7d21-41b6-932f-bb76eb462178"
pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,  # Use the dimension we found
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print("Index created")
else:
    print("Index already exists")

# Store vectors in DB
vectorstore = PineconeVectorStore.from_documents(
    docs,
    ollama_emb,
    index_name=index_name      
)

print("Vectors stored in Pinecone")
print(vectorstore)
