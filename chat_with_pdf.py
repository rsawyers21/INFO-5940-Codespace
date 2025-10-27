import streamlit as st
import os
import pypdf
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

# Set environment variables
os.environ['OPENAI_API_KEY'] = os.environ.get("API_KEY")
os.environ['OPENAI_BASE_URL'] = "https://api.ai.it.cornell.edu"

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
)

st.title("📝 RAG-Enhanced File Q&A with Langchain")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload documents and ask me questions about them!"}
    ]

# Define State for LangGraph (following notebook pattern)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create document chunks"""
    # Following notebook's chunking pattern
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # larger chunks for better context
        chunk_overlap=200 # arbitrary overlap
    )
    
    all_documents = []
    
    for uploaded_file in uploaded_files:
        # Extract text based on file type
        if uploaded_file.type == "text/plain":  # .txt files
            content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":  # .pdf files
            content = extract_text_from_pdf(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            continue
        
        # Create document object
        doc = Document(page_content=content, metadata={"source": uploaded_file.name})
        all_documents.append(doc)
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(all_documents)
    return chunks

def create_rag_graph(vectorstore):
    """Create LangGraph RAG workflow following notebook pattern"""
    
    # Define the prompt template (from notebook)
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    
    Context: {context} 
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    # Define workflow functions (from notebook)
    def retrieve(state: State):
        retrieved_docs = vectorstore.similarity_search(state["question"], k=20)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    # Build the graph (from notebook)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph

# Process PDFs
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# File upload interface
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=("txt", "pdf"), 
    accept_multiple_files=True,
    help="Upload .txt or .md files to chat with them"
)

# Process new files
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        with st.spinner(f"Processing {len(new_files)} new documents..."):
            try:
                # Process documents into chunks
                chunks = process_uploaded_files(new_files)
                
                # Create or update vector store (following notebook pattern)
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=chunks, 
                        embedding=OpenAIEmbeddings(model="openai.text-embedding-3-large")
                    )
                else:
                    st.session_state.vectorstore.add_documents(chunks)
                
                # Create RAG graph
                st.session_state.rag_graph = create_rag_graph(st.session_state.vectorstore)
                
                # Mark files as processed
                for f in new_files:
                    st.session_state.processed_files.add(f.name)
                
                st.success(f"✅ Processed {len(new_files)} new documents with {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

# Display current uploaded files
if st.session_state.processed_files:
    with st.expander("📁 Uploaded Files"):
        for filename in st.session_state.processed_files:
            st.write(f"• {filename}")

# Chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
question = st.chat_input(
    "Ask something about your documents",
    disabled=st.session_state.rag_graph is None,
)

if question and st.session_state.rag_graph:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use LangGraph RAG workflow (following notebook pattern)
                result = st.session_state.rag_graph.invoke({"question": question})
                
                answer = result["answer"]
                context_docs = result["context"]
                
                # Display the answer
                st.write(answer)
                
                # Show sources in an expander
                with st.expander("📄 Sources"):
                    for i, doc in enumerate(context_docs[:5], 1):  # Show top 5 sources
                        source = doc.metadata.get("source", "Unknown source")
                        st.write(f"**[{i}] {source}**")
                        st.write(f"_{doc.page_content[:200]}..._")
                        st.write("---")
                
            except Exception as e:
                answer = f"Sorry, I encountered an error: {str(e)}"
                st.error(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. **Upload documents**: Use the file uploader to add .txt or .pdf files
    2. **Wait for processing**: Files will be chunked and indexed with embeddings
    3. **Ask questions**: Use the chat interface to ask about your documents
    4. **View sources**: Click on 'Sources' to see which document chunks were used
    """)
    
    if st.session_state.rag_graph:
        st.success("✅ RAG Pipeline ready!")
    else:
        st.info("⏳ Upload files to get started")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload documents and ask me questions about them!"}
        ]
        st.rerun()