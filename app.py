import os
import tempfile
import streamlit as st
import chromadb
from pypdf import PdfReader  # Lightweight replacement for PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- 1. CONFIGURATION & UI LAYOUT ---
st.set_page_config(page_title="Kaggle-Ollama RAG App", layout="wide")
st.title("📚 Streamlit RAG connected to Kaggle Ollama & Chroma Cloud")

st.sidebar.header("Connection Settings")
KAGGLE_NGROK_URL = st.sidebar.text_input(
    "Enter Kaggle Ngrok URL:", 
    value="https://evident-lens-surpass.ngrok-free.dev",
    placeholder="https://evident-lens-surpass.ngrok-free.dev"
)
SELECTED_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# --- 2. DYNAMIC OLLAMA EMBEDDINGS ---
def get_ollama_embeddings(base_url):
    return OllamaEmbeddings(
        base_url=base_url,
        model=EMBEDDING_MODEL
    )

# --- 3. CHROMA CLOUD INITIALIZATION ---
# Removed @st.cache_resource to prevent memory accumulation in Render's tight environment
def get_chroma_cloud_vectorstore(base_url):
    """
    Connects to Chroma Cloud using CloudClient parameters.
    """
    chroma_client = chromadb.CloudClient(
        api_key='ck-2wXzKWc4NjXhpb2QGfv8tJeC7J4s5GMApkjtJyo5wGDj',
        tenant='8322a478-cf18-4473-a3ef-d683ef1e9434',
        database='myVectorDB'
    )
    
    embedding_fn = get_ollama_embeddings(base_url)
    
    return Chroma(
        client=chroma_client,
        collection_name="rag_documents_collection",
        embedding_function=embedding_fn
    )

# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload a PDF document to query", type=["pdf"])

# --- 5. LOW-MEMORY DOCUMENT PROCESSING & INGESTION ---
if uploaded_file is not None:
    if not KAGGLE_NGROK_URL:
        st.error("❌ Please provide a valid Ngrok URL in the sidebar before processing documents.")
    else:
        if "document_indexed" not in st.session_state:
            with st.spinner("Processing document efficiently..."):
                try:
                    # Initialize Cloud Vector Store
                    vector_store = get_chroma_cloud_vectorstore(KAGGLE_NGROK_URL)
                    
                    # Memory-efficient Text Splitter
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    
                    # Read PDF using pypdf instead of heavy LangChain loaders
                    reader = PdfReader(uploaded_file)
                    
                    # Stream chunks page-by-page directly to Chroma to avoid massive lists in RAM
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Split text for just this single page
                            chunks = text_splitter.split_text(page_text)
                            
                            # Add metadata so Langchain reads it correctly
                            metadata = [{"source": uploaded_file.name, "page": i + 1} for _ in chunks]
                            
                            # Push immediately to Chroma Cloud, then empty memory
                            vector_store.add_texts(texts=chunks, metadatas=metadata)
                    
                    st.session_state.document_indexed = True
                    st.success("🎉 Document successfully embedded and indexed into Chroma Cloud!")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# --- 6. QA INTERFACE & INFERENCE LOOP ---
if KAGGLE_NGROK_URL:
    st.write("---")
    user_query = st.text_input("Ask something about your document:")
    
    if user_query:
        with st.spinner("Searching cloud vectors and querying hosted Llama 3.2 model..."):
            try:
                vector_store = get_chroma_cloud_vectorstore(KAGGLE_NGROK_URL)
                llm = Ollama(base_url=KAGGLE_NGROK_URL, model=SELECTED_MODEL)
                
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_query)
                
                if not relevant_docs:
                    st.warning("⚠️ No relevant matches found in Chroma Cloud database.")
                    context = ""
                else:
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                rag_prompt = f"""
                You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, say that you don't know.
                
                Context:
                {context}
                
                Question: 
                {user_query}
                
                Answer:
                """
                
                response = llm.invoke(rag_prompt)
                st.markdown("### 🤖 Answer:")
                st.write(response)
                
                if relevant_docs:
                    with st.expander("View Retrieved Source Chunks (Chroma Cloud)"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content)
                            
            except Exception as e:
                st.error(f"Communication breakdown. Error: {e}")
