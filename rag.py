import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Load the PDF document
pdf_path = os.path.join(os.getcwd(), "assets", "PDF_FILE_NAME")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()[0:100] # Start with first 100 pages

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
print("Splitting documents...")
chunks = list(tqdm(text_splitter.split_documents(documents), desc="Processing chunks"))
print(f"Created {len(chunks)} chunks. Generating embeddings...")

# 3. Create embeddings and store in ChromaDB
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if embeddings already exist
if os.path.exists("./chroma_db"):
    print("Loading existing vector database...")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
else:
    print("Creating new vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    print("Vector database created and persisted.")

# 4. Create a retrieval chain
# Increase k to get more context
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Create a custom prompt template that gives better instructions to the model
prompt_template = """
###EXAMPLE PROMPT###

You are an expert on the National Electrical Code (NEC). Use the following context from the NFPA 70 NEC 2023 document to answer the question accurately and with specific details.

Context:
{context}

Question:
{question}

When answering, focus on providing exact information from the NEC including any numerical values, specific article references, or quoted definitions. If the answer is not directly in the context, say "I don't see a specific definition for this in the provided context." If there is an exact definition, provide it verbatim in quotes first, then explain if needed.

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Use Ollama with Llama 3
try:
    print("Connecting to Ollama with llama3...")
    llm = Ollama(model="llama3", temperature=0.1)
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("Make sure Ollama is running and llama3 model is downloaded.")
    exit(1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 5. Interactive query interface
def ask_question(qa_chain):
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        print("\nSearching for answer...")
        try:
            result = qa_chain.invoke({"query": query})
            print("\nAnswer:", result["result"])

            # Print sources for verification
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"][:3]):
                print(f"Document {i+1}:")
                print(f"Page {doc.metadata.get('page', 'unknown')}: {doc.page_content[:300]}...")
                print()
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again with a different question.")

# Run the interactive interface
if __name__ == "__main__":
    print("RAG System initialized.")
    ask_question(qa_chain)
