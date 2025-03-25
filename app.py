import streamlit as st
#import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os

# Initialize Streamlit App
st.set_page_config(page_title="Travelling Agent Chatbot", layout="wide")
st.title("✈️ Travelling Agent Chatbot")
st.markdown("Your AI-powered travel assistant for planning trips, budgeting, and finding destinations.")

# Initialize memory and models
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChatGroq correctly
try:
    chat = ChatGroq(
        model_name="llama3-70b-8192", 
        temperature=0.7, 
        groq_api_key="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"
    )
except Exception as e:
    st.error(f"⚠️ Chat model initialization failed: {str(e)}")
    chat = None  # Prevents further errors

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

def retrieve_context(query, top_k=1):
    """Retrieve relevant documents from ChromaDB using embeddings."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results and results["documents"] else ["No relevant context found."]

def query_llama3(user_query):
    """Query the Llama3 model with user input and retrieved context."""
    system_prompt = """
    You are 'Travelling Agent,' an intelligent travel assistant chatbot. Your role is to help users plan trips, 
    find destinations, suggest transport options, and provide cost estimates.
    """
    
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"DB Context: {retrieved_context}\nQuestion: {user_query}")
    ]
    
    if not isinstance(chat, ChatGroq):  
        return user_query, "⚠️ Chat model initialization error."

    try:
        response = chat.invoke(messages)
        return user_query, response.content
    except Exception as e:
        return user_query, f"⚠️ API Error: {str(e)}"

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for chat_msg in st.session_state.chat_history:
    user_msg, bot_response = chat_msg
    st.markdown(f'<div style="background-color:#C599B6;padding:10px;border-radius:10px;margin:5px;align-self:flex-start;">👤 {user_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background-color:#C599B6;padding:10px;border-radius:10px;margin:5px;align-self:flex-end;">🤖 {bot_response}</div>', unsafe_allow_html=True)

# User input box at the bottom
user_query = st.chat_input("Type your message...")
if user_query:
    user_msg, bot_response = query_llama3(user_query)
    st.session_state.chat_history.append((user_msg, bot_response))
    st.rerun()  # Refresh the app to update chat history

# Clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!")
    st.rerun()
