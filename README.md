# RAG Project: Local PDF Question-Answering System

A powerful Retrieval-Augmented Generation (RAG) system that allows you to ask questions about technical PDFs without needing any specialized knowledge. This project uses:

- **LangChain**: For the document processing pipeline
- **ChromaDB**: As the vector database to store and retrieve document chunks
- **HuggingFace Embeddings**: To create vector representations of text
- **Ollama with Llama 3**: For generating answers locally without API costs

## Project Overview

This system allows you to load a PDF document, ask questions about it, and get detailed answers with source citations - all running 100% locally. The project includes a sample electrical code PDF, but you can easily modify it to work with any technical document.

## Prerequisites

- Python 3.8 or higher
- Ollama (for running Llama 3 locally)
- At least 8GB of RAM (16GB recommended)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-project.git
cd rag-project
```

2. Create a virtual environment:
```bash
python -m venv rag_project
source rag_project/bin/activate  # On Windows: rag_project\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install langchain langchain-community chromadb pypdf sentence-transformers transformers torch tqdm
```

4. Install Ollama:
   - **macOS**: Download from [https://ollama.com/download/mac](https://ollama.com/download/mac)
   - **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
   - **Windows**: Download from [https://ollama.com/download/windows](https://ollama.com/download/windows)

5. Pull the Llama 3 model:
   - Start Ollama (open the application after installing it)
   - Open a terminal and run: `ollama run llama3`
   - After it downloads, you can exit (type `/bye`)

## Running the Application

1. Place your PDF file in the `assets` folder.

2. Modify the file path in `rag.py` if needed:
```python
pdf_path = os.path.join(os.getcwd(), "assets", "your-pdf-file.pdf")
```

3. Run the application:
```bash
python rag.py
```

4. The system will:
   - Load the first 100 pages of your PDF (configurable)
   - Split it into chunks
   - Create and store embeddings in ChromaDB
   - Start an interactive interface where you can ask questions

## Usage Examples

When prompted, you can ask questions about the document.

The system will retrieve relevant sections from the document and generate an answer, then show you the source information.

Type 'exit' to quit the application.

## How It Works

1. **Document Loading**: Loads the PDF and extracts text.
2. **Text Splitting**: Breaks the document into manageable chunks.
3. **Embedding Creation**: Converts text chunks into vector embeddings using HuggingFace's models.
4. **Storage**: Stores these embeddings in ChromaDB for efficient retrieval.
5. **Query Processing**:
   - When you ask a question, it's converted to an embedding.
   - ChromaDB finds the most similar document chunks.
   - The original question and relevant chunks are sent to Llama 3.
   - Llama 3 generates an answer based on the context provided.
   - The system shows you both the answer and the source chunks.

## Customization

You can adjust the following parameters in the code:

- `documents = loader.load()[0:100]`: Change how many pages to process
- `chunk_size=2000, chunk_overlap=200`: Adjust chunking parameters
- `search_kwargs={"k": 5}`: Control how many chunks are retrieved
- `temperature=0.1`: Adjust the creativity level of responses
- Edit the prompt template for different instruction styles

## Troubleshooting

- **"Command not found: ollama"**: Make sure Ollama is installed and in your PATH.
- **"Error connecting to Ollama"**: Ensure the Ollama application is running.
- **"File not found"**: Check that your PDF is in the correct location.
- **Memory errors**: Reduce the number of pages processed or increase chunk size.

## License

MIT

