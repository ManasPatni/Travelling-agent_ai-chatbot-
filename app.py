import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os

# Optional: Set offline mode if needed
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set page config
st.set_page_config(page_title="✈️ Travelling Agent Chatbot", layout="wide")
st.title("✈️ Travelling Agent Chatbot")
st.markdown("Your AI-powered travel assistant for planning trips, budgeting, and finding destinations.")

# === Initialize memory and embeddings ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"⚠️ Failed to load embedding models: {str(e)}")
    embedding_model = None
    semantic_model = None

# === Initialize ChatGroq model ===
try:
    chat = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY", "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")
    )
except Exception as e:
    st.error(f"⚠️ Chat model initialization failed: {str(e)}")
    chat = None

# === Initialize ChromaDB ===
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")
except Exception as e:
    st.error(f"⚠️ Failed to initialize ChromaDB: {str(e)}")
    collection = None

def retrieve_context(query, top_k=1):
    """Retrieve relevant documents from ChromaDB using embeddings."""
    if not embedding_model or not collection:
        return ["Embedding model or ChromaDB not initialized."]
    
    try:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0] if results and results["documents"] else ["No relevant context found."]
    except Exception as e:
        return [f"⚠️ Retrieval Error: {str(e)}"]

def query_llama3(user_query):
    """Query the Llama3 model with user input and retrieved context."""
    if not chat:
        return user_query, "⚠️ Chat model initialization error."

    system_prompt = (
        "You are 'Travelling Agent,' an intelligent travel assistant chatbot. Your role is to help users "
        "plan trips, find destinations, suggest transport options, and provide cost estimates."
    )
    
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"DB Context: {retrieved_context}\nQuestion: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        return user_query, response.content
    except Exception as e:
        return user_query, f"⚠️ API Error: {str(e)}"

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Display chat messages ===
for user_msg, bot_response in st.session_state.chat_history:
    st.markdown(f'<div style="background-color:#C599B6;padding:10px;border-radius:10px;margin:5px;">👤 {user_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background-color:#F1C6E7;padding:10px;border-radius:10px;margin:5px;">🤖 {bot_response}</div>', unsafe_allow_html=True)

# === Input box ===
user_query = st.chat_input("Type your message...")
if user_query:
    user_msg, bot_response = query_llama3(user_query)
    st.session_state.chat_history.append((user_msg, bot_response))
    st.rerun()

# === Sidebar actions ===
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!")
    st.rerun()
