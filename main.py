import os
from typing import List
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from your local .env file
load_dotenv()

# ==========================================
# CHECKPOINT 1: Document ingestion + chunking
# ==========================================
def load_and_chunk_documents(file_paths: List[str]) -> List[Document]:
    raw_documents = []
    
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        raw_documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
    
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}_chunk_{idx}"

    return chunks

# ==========================================
# CHECKPOINT 2: Embedding generation
# ==========================================
def get_embedding_model():
    # Make sure OPENAI_API_KEY is in your .env file
    return OpenAIEmbeddings(model="text-embedding-3-small")

# ==========================================
# CHECKPOINT 3: Vector storage + similarity search
# ==========================================
def build_vector_store(chunks: List[Document], persist_directory: str = "./chroma_db"):
    embeddings = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_db

# ==========================================
# CHECKPOINT 4: Retrieval + prompt setup
# ==========================================
RAG_PROMPT_TEMPLATE = """\
You are an assistant for question-answering tasks. 
Use ONLY the provided retrieved context to answer the question. 
If you do not know the answer or if the context does NOT contain enough information to answer, state clearly: "I cannot answer this question based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

def format_context_with_metadata(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        source_name = d.metadata.get("source", "Unknown")
        chunk_id = d.metadata.get("chunk_id", "N/A")
        formatted.append(f"[Source: {source_name} | Chunk: {chunk_id}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

# ==========================================
# CHECKPOINT 5: Response generation with source reference
# ==========================================
def generate_rag_response(vector_db: Chroma, query: str):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    
    formatted_context = format_context_with_metadata(retrieved_docs)
    
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    chain = prompt | llm | StrOutputParser()
    
    raw_response = chain.invoke({
        "context": formatted_context,
        "question": query
    })
    
    sources = [
        f"Doc: {d.metadata.get('source', 'Unknown')}, Chunk: {d.metadata.get('chunk_id', 'N/A')}" 
        for d in retrieved_docs
    ]
    
    return {
        "query": query,
        "response": raw_response,
        "sources_used": list(set(sources))
    }

# ==========================================
# CHECKPOINT 6 & EXECUTION: Running the pipeline
# ==========================================
if __name__ == "__main__":
    # 1. Provide a sample document path (make sure you create this file/folder)
    sample_doc_path = ["sample_docs/test_document.txt"]
    
    # 2. Run Ingestion & Chunking
    print("Chunking documents...")
    chunks = load_and_chunk_documents(sample_doc_path)
    
    # 3. Build Vector DB
    print("Building Vector Database...")
    vector_db = build_vector_store(chunks)
    
    # 4. Test Normal Query
    print("\n--- TEST 1: IN-DOCUMENT QUERY ---")
    valid_query = "What is the secret deployment authorization code required to switch the DNS records?"
    result = generate_rag_response(vector_db, valid_query)
    print(f"Q: {result['query']}\nA: {result['response']}\nSources: {result['sources_used']}")
    
    # 5. Test Hallucination Guardrail (Checkpoint 6 Requirement)
    print("\n--- TEST 2: NOT IN DOCS (Hallucination Check) ---")
    out_of_scope_query = "What is the capital of Mars?"
    guardrail_result = generate_rag_response(vector_db, out_of_scope_query)
    print(f"Q: {guardrail_result['query']}\nA: {guardrail_result['response']}\nSources: {guardrail_result['sources_used']}")
