import os
import tempfile
import streamlit as st
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- 1. CONFIGURATION & UI LAYOUT ---
st.set_page_config(page_title="Kaggle-Ollama RAG App", layout="wide")
st.title("📚 Streamlit RAG connected to Kaggle Ollama & Chroma Cloud")

# Sidebar for Ngrok endpoint configuration
st.sidebar.header("Connection Settings")
KAGGLE_NGROK_URL = st.sidebar.text_input(
    "Enter Kaggle Ngrok URL:", 
    value="https://evident-lens-surpass.ngrok-free.dev",
    placeholder="https://evident-lens-surpass.ngrok-free.dev"
)
SELECTED_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# --- 2. DYNAMIC OLLAMA EMBEDDINGS ---
# We initialize embeddings dynamically inside a function so it updates if the URL changes
def get_ollama_embeddings(base_url):
    return OllamaEmbeddings(
        base_url=base_url,
        model=EMBEDDING_MODEL
    )

# --- 3. CHROMA CLOUD INITIALIZATION ---
@st.cache_resource
def get_chroma_cloud_vectorstore(base_url):
    """
    Connects to Chroma Cloud using CloudClient parameters and wraps it 
    with LangChain's Chroma vector store using the remote embedding model.
    """
    # 1. Initialize native Chroma CloudClient
    chroma_client = chromadb.CloudClient(
        api_key='ck-2wXzKWc4NjXhpb2QGfv8tJeC7J4s5GMApkjtJyo5wGDj',
        tenant='8322a478-cf18-4473-a3ef-d683ef1e9434',
        database='myVectorDB'
    )
    
    # 2. Grab our dynamic embedding function
    embedding_fn = get_ollama_embeddings(base_url)
    
    # 3. Bind to LangChain's wrapper interface using the active client connection
    vector_store = Chroma(
        client=chroma_client,
        collection_name="rag_documents_collection",
        embedding_function=embedding_fn
    )
    return vector_store

# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload a PDF document to query", type=["pdf"])

# --- 5. DOCUMENT PROCESSING & INGESTION ---
if uploaded_file is not None:
    if not KAGGLE_NGROK_URL:
        st.error("❌ Please provide a valid Ngrok URL in the sidebar before processing documents.")
    else:
        # Check if we have already indexed the document to prevent infinite re-indexing loops
        if "document_indexed" not in st.session_state:
            with st.spinner("Processing document... Generating vectors via hosted nomic-embed-text and pushing to Chroma Cloud."):
                
                # Securely write uploaded file to temp directory
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    # Initialize Cloud Vector Store
                    vector_store = get_chroma_cloud_vectorstore(KAGGLE_NGROK_URL)
                    
                    # Load and chunk PDF
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    splits = text_splitter.split_documents(docs)
                    
                    # Add documents directly to our remote cloud collection instance
                    vector_store.add_documents(documents=splits)
                    
                    st.session_state.document_indexed = True
                    st.success("🎉 Document successfully embedded and indexed into Chroma Cloud!")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

# --- 6. QA INTERFACE & INFERENCE LOOP ---
if KAGGLE_NGROK_URL:
    st.write("---")
    user_query = st.text_input("Ask something about your document:")
    
    if user_query:
        with st.spinner("Searching cloud vectors and querying hosted Llama 3.2 model..."):
            try:
                # 1. Establish connections dynamically using current parameters
                vector_store = get_chroma_cloud_vectorstore(KAGGLE_NGROK_URL)
                llm = Ollama(base_url=KAGGLE_NGROK_URL, model=SELECTED_MODEL)
                
                # 2. Retrieve top 3 relevant chunks from Chroma Cloud
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_query)
                
                if not relevant_docs:
                    st.warning("⚠️ No relevant matches found in Chroma Cloud database.")
                    context = ""
                else:
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # 3. Standard RAG prompt assembly
                rag_prompt = f"""
                You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, say that you don't know.
                
                Context:
                {context}
                
                Question: 
                {user_query}
                
                Answer:
                """
                
                # 4. Generate answer from Ollama endpoint
                response = llm.invoke(rag_prompt)
                st.markdown("### 🤖 Answer:")
                st.write(response)
                
                if relevant_docs:
                    with st.expander("View Retrieved Source Chunks (Chroma Cloud)"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content)
                        
            except Exception as e:
                st.error(f"Communication breakdown. Verify your public Ngrok URL tunnel and Chroma cloud access credentials. Error: {e}")
