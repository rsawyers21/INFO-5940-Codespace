import streamlit as st
import os
import pypdf
import uuid
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from bs4 import BeautifulSoup

# Set environment variables
os.environ['OPENAI_API_KEY'] = os.environ.get("API_KEY")
os.environ['OPENAI_BASE_URL'] = "https://api.ai.it.cornell.edu"

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
)

st.title("RAG-Enhanced Q&A Chatbot")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = {}   # Store chunks per file for removal
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload documents and ask me questions about them!"}
    ]

# Define state for LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Clean text from PDFs
def clean_extracted_text(text):
    if not text:
        return text
    
    # Remove excessive whitespace and fix common extraction issues
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)  # Join broken words
    text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)  # Join sentences
    text = re.sub(r'([a-z,])\s*\n\s*([a-z])', r'\1 \2', text)  # Join broken lines
    
    # Fix bullet points and lists
    text = re.sub(r'\n\s*[•·]\s*', '\n• ', text)
    text = re.sub(r'\n\s*[-]\s*', '\n- ', text)
    
    # Clean up HTML artifacts that might remain
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r'<[^>]+>', '', text)  # Remove any remaining HTML tags
    
    # Clean up multiple newlines but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

# Process uploaded files and chunk documents
def process_uploaded_files(uploaded_files):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Slightly smaller chunks for better context
        chunk_overlap=100,   # Small overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""],    # Separate by paragraphs, sentences, then words
        length_function=len
    )
    
    file_chunks = {}
    
    for uploaded_file in uploaded_files:
        # Extract text based on file type
        if uploaded_file.type == "text/plain":  # .txt files
            content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":  # .pdf files
            content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/html":  # .html files
            content = extract_text_from_html(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            continue
        
        # Clean the extracted text
        content = clean_extracted_text(content)

        # Skip empty content
        if not content.strip():
            st.warning(f"No text found in {uploaded_file.name}")
            continue

        # Create document object with enhanced metadata
        doc = Document(
            page_content=content, 
            metadata={
                "source": uploaded_file.name,
                "file_type": uploaded_file.type,
                "content_length": len(content)
            }
        )
        
        # Split document into chunks
        chunks = text_splitter.split_documents([doc])
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"{uploaded_file.name}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        file_chunks[uploaded_file.name] = chunks
    
    return file_chunks

# Rebuild vector store after file addition or removal
def rebuild_vectorstore():
    # Always clear existing vector store and RAG graph
    st.session_state.vectorstore = None
    st.session_state.rag_graph = None

    if not st.session_state.file_chunks:
        return
    
    # Collect all chunks from remaining files
    all_chunks = []
    for chunks in st.session_state.file_chunks.values():
        all_chunks.extend(chunks)
    
    if all_chunks:
        # Create new vector store with uuid
        collection_name = f"documents_{uuid.uuid4().hex[:8]}"
        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=OpenAIEmbeddings(model="openai.text-embedding-3-large"),
            collection_name=collection_name
        )
        # Recreate RAG graph
        st.session_state.rag_graph = create_rag_graph(st.session_state.vectorstore)
    else:
        st.session_state.vectorstore = None
        st.session_state.rag_graph = None

# Hanel file additions and removals
def handle_file_changes(current_uploaded_files):
    current_filenames = {f.name for f in current_uploaded_files} if current_uploaded_files else set()
    
    # Detect removed files
    removed_files = st.session_state.processed_files - current_filenames
    if removed_files:
        # Remove chunks for deleted files
        for filename in removed_files:
            if filename in st.session_state.file_chunks:
                del st.session_state.file_chunks[filename]
        # Update processed files set
        st.session_state.processed_files = current_filenames & st.session_state.processed_files
        # Rebuild vector store without removed files
        with st.spinner(f"Removing {len(removed_files)} file(s) from knowledge base..."):
            rebuild_vectorstore()
        
        st.success(f"Removed {len(removed_files)} file(s): {', '.join(removed_files)}")
    
    # Detect new files
    new_files = [f for f in current_uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        with st.spinner(f"Processing {len(new_files)} new document(s)..."):
            try:
                # Process new documents into chunks
                new_file_chunks = process_uploaded_files(new_files)
                # Add new chunks to our storage
                st.session_state.file_chunks.update(new_file_chunks)
                # Rebuild vector store with all remaining files
                rebuild_vectorstore()
                # Mark files as processed
                for f in new_files:
                    st.session_state.processed_files.add(f.name)
                
                st.success(f"Processed {len(new_files)} new document(s)")
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

# Create RAG graph
def create_rag_graph(vectorstore):
    template = """
    You are a helpful assistant that answers questions based on the provided context documents. ONLY use the information provided in the context below
    to answer the question. If the context does not contain the answer, say that you are unable to answer. If there is conflicting information, mention it. Use three
    sentences maximum and keep the answer concise. 

    Question: {question} 
    
    Context: {context} 
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    def retrieve(state: State):
        retrieved_docs = vectorstore.similarity_search(state["question"], k=8)
        return {"context": retrieved_docs}

    def generate(state: State):
        formatted_context = ""
        for i, doc in enumerate(state["context"], 1):
            source = doc.metadata.get("source", "Unknown source")
            chunk_info = ""
            if "chunk_index" in doc.metadata:
                chunk_info = f" (chunk {doc.metadata['chunk_index'] + 1})"
            
            formatted_context += f"\n--- Document {i}: {source}{chunk_info} ---\n"
            formatted_context += doc.page_content + "\n"
        
        # Generate response with enhanced context
        messages = prompt.invoke({
            "question": state["question"], 
            "context": formatted_context.strip()
        })
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph

# Process PDFs
def extract_text_from_pdf(uploaded_file):
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Process HTML
def extract_text_from_html(uploaded_file):
    html_content = uploaded_file.read().decode("utf-8")
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    return text

# File upload interface
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=("txt", "pdf", "html"), 
    accept_multiple_files=True,
    help="Upload .txt, .pdf, or .html files to chat with them"
)

handle_file_changes(uploaded_files)

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
                # Use LangGraph RAG workflow
                result = st.session_state.rag_graph.invoke({"question": question})
                
                answer = result["answer"]
                context_docs = result["context"]
                
                # Display the answer
                st.write(answer)
                
                # Show sources in an expander
                with st.expander("Sources"):
                    for i, doc in enumerate(context_docs[:5], 1):  # Show top 5 sources
                        source = doc.metadata.get("source", "Unknown source")
                        st.write(f"**[{i}] {source}**")
                        st.write(f"{doc.page_content}...")
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
    1. **Upload documents**: Use the file uploader to add .txt, .pdf, or .html files
    2. **Wait for processing**: Files will be chunked and indexed with embeddings
    3. **Remove files**: Click 'X' next to any file to remove it from the knowledge base
    4. **Ask questions**: Use the chat interface to ask about your documents
    5. **View sources**: Click on 'Sources' to see which document chunks were used
    """)
    
    if st.session_state.rag_graph:
        st.success("RAG Pipeline ready!")
    else:
        st.info("Upload files to get started")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload documents and ask me questions about them!"}
        ]
        st.rerun()