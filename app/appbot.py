import streamlit as st
from langchain_community.document_loaders import JSONLoader

def load_json_documents(file_path):
    """
    Load documents from a JSON file using JSONLoader.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        List[Document]: List of loaded LangChain Document objects.
    """
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[] | {text: (.title + \": \" + .description)}",
        text_content=False
    )
    docs = loader.load()
    print("Documents loaded successfully.")
    return docs

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        docs (List[Document]): List of Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        
    Returns:
        List[Document]: List of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Number of chunks after splitting: {len(split_docs)}")
    return split_docs

from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get the embedding model used for encoding documents and queries.
    
    Args:
        model_name (str): Name or path of the Hugging Face embedding model.
        
    Returns:
        Embeddings: HuggingFaceEmbeddings object.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

from langchain_community.vectorstores import Chroma

def create_vectorstore(documents, embedding, persist_directory="../vector_store", collection_name="hr_policy_collection"):
    """
    Create and persist a Chroma vector store from documents.
    
    Args:
        documents (List[Document]): List of document chunks.
        embedding (Embeddings): Embedding model.
        persist_directory (str): Directory to save the vector store.
        collection_name (str): Name of the collection.
        
    Returns:
        Chroma: The created vector store.
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Vector store created successfully.")
    return vectorstore


def get_retriever(vectorstore, search_type="mmr", k=5, lambda_mult=0.5):
    """
    Get a configured retriever from the vector store.
    
    Args:
        vectorstore (Chroma): Vector store instance.
        search_type (str): Type of search ('similarity', 'mmr').
        k (int): Number of documents to retrieve.
        lambda_mult (float): Controls diversity in MMR search.
        
    Returns:
        Retriever: Configured retriever object.
    """
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "lambda_mult": lambda_mult}
    )
    return retriever

from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(retriever, llm_model="llama-3.1-8b-instant", temperature=0.1):
    """
    Create a question-answering chain using LLM and retriever.
    
    Args:
        retriever (Retriever): The retriever object.
        llm_model (str): Model name for ChatGroq.
        temperature (float): Temperature for LLM generation.
        
    Returns:
        RunnableSequence: QA chain ready to use.
    """
    llm = ChatGroq(model=llm_model, temperature=temperature)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an HR policy assistant. Use the context below to answer the user's question.
If the question is a casual greeting (such as "hi", "hello", "hey", "good morning", etc.), respond with a friendly greeting as an HR assistant.
If the answer is not found in the context, reply with: "Sorry, I couldn't find that in the HR policy."

Context:
{context}

Question: {question}

Answer:
"""
    )

    chain = RunnableSequence(
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        },
        prompt_template,
        llm,
        StrOutputParser()
    )
    return chain


# Page configuration
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="💼",
    layout="wide"
)

# Simple CSS for clean UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        background-color: #ffffff;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background-color: #f8f9fa;
        border-left: 3px solid #666666;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff;
        border-left: 3px solid #333333;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #cccccc;
        padding: 0.5rem;
        background-color: #ffffff;
        color: #333333;
    }
    
    .stButton > button:hover {
        background-color: #f8f9fa;
        border-color: #999999;
    }
    
    .main-title {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session State Initialization
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------------------
# Sidebar - App Info & Instructions
# ---------------------------
with st.sidebar:
    st.header("📋 Information")
    
    # Status
    st.subheader("Status")
    if st.session_state.qa_chain:
        st.success("🟢 Ready")
    else:
        st.warning("🟡 Loading...")
    
    # Instructions
    st.subheader("How to use")
    st.info("""
    Welcome to your AI-powered HR Policy Assistant!
    
    Simply type your question about HR policies in the chat below.
    
    The assistant will:
    • Search through policy documents
    • Provide accurate information
    • Answer your specific questions
    
    💡 Powered by RAG (Retrieval-Augmented Generation)
    """)
    
    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history.clear()
        st.rerun()

# ---------------------------
# Load QA Chain
# ---------------------------
@st.cache_resource
def load_qa_chain():
    docs = load_json_documents("../data/hr_policy.json")
    split_docs = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(split_docs, embeddings)
    retriever = get_retriever(vectorstore)
    return create_qa_chain(retriever)

# ---------------------------
# Main App Layout
# ---------------------------
st.markdown('<div class="main-title">', unsafe_allow_html=True)
st.title("💼 HR Policy Chatbot")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">', unsafe_allow_html=True)
st.markdown("Ask anything about company policies — I'm here to help!")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize QA chain if not already done
with st.spinner("🧠 Loading the HR knowledge base..."):
    if not st.session_state.qa_chain:
        st.session_state.qa_chain = load_qa_chain()

# ---------------------------
# Render Chat History
# ---------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# ---------------------------
# User Input Handling
# ---------------------------
user_input = st.chat_input("Type your HR policy question...")
if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking..."):
            response = st.session_state.qa_chain.invoke(user_input)
        st.markdown(response)
        
        # Save bot response
    st.session_state.chat_history.append({"role": "assistant", "text": response})