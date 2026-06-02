import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Kaggle-Ollama RAG App", layout="wide")
st.title("📚 Streamlit RAG connected to Kaggle Ollama")

# Sidebar for Ngrok endpoint configuration
st.sidebar.header("Connection Settings")
KAGGLE_NGROK_URL = st.sidebar.text_input(
    "Enter Kaggle Ngrok URL:", 
    value="",
    placeholder="https://evident-lens-surpass.ngrok-free.dev"
)
SELECTED_MODEL = "llama3.2"

# --- 2. INITIALIZE EMBEDDINGS & LLM ---
# We cache the embedding model so Render doesn't reload it on every user click
@st.cache_resource
def load_local_embeddings():
    # Uses a highly efficient, small local model to create vectors on Render's CPU
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if KAGGLE_NGROK_URL:
    llm = Ollama(base_url=KAGGLE_NGROK_URL, model=SELECTED_MODEL)
    embeddings = load_local_embeddings()
else:
    st.info("💡 Please enter your live Kaggle Ngrok public URL in the sidebar to connect to the LLM.")
    st.stop()

# --- 3. FILE UPLOADER & PROCESSING ---
uploaded_file = st.file_uploader("Upload a PDF document to query", type=["pdf"])

if uploaded_file is not None and "vector_store" not in st.session_state:
    with st.spinner("Processing document... Parsing and embedding locally on Render."):
        # Securely write uploaded file to Render's temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load and chunk PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            
            # Create an in-memory Chroma database
            vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vector_store = vector_store
            st.success("🎉 Document indexed successfully! Ask your questions below.")
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# --- 4. QA INTERFACE ---
if "vector_store" in st.session_state:
    st.write("---")
    user_query = st.text_input("Ask something about your document:")
    
    if user_query:
        with st.spinner("Searching document and querying Kaggle LLM..."):
            # Retrieve top 3 relevant chunks
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(user_query)
            
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
            
            try:
                # Direct API call to Kaggle via Ngrok
                response = llm.invoke(rag_prompt)
                st.markdown("### 🤖 Answer:")
                st.write(response)
                
                with st.expander("View Retrieved Source Chunks"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.caption(doc.page_content)
            except Exception as e:
                st.error(f"Failed to communicate with Kaggle Ollama instance. Check your Ngrok URL. Error: {e}")
