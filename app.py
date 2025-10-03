import os 
import re
import requests
import pandas as pd
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv

from langchain.schema import Document 
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Page config
st.set_page_config(page_title="Research Paper Reader", page_icon="📚", layout="wide")

# Load environment
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize session state
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=2
    )
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'paper_loaded' not in st.session_state:
    st.session_state.paper_loaded = False


# --- Embeddings ---
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )


# Helper functions
def strip_references_regex(text):
    split_text = re.split(r"\nReferences\n|\nREFERENCES\n|\nBibliography\n", text, flags=re.IGNORECASE)
    main_text = split_text[0]
    lines = main_text.split("\n")
    filtered_lines = []
    for line in lines:
        if re.match(r"\[\d+\]", line.strip()):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def load_paper(title):
    try:
        df = pd.read_csv("papers_title.csv")
        results = df[df['title'].str.contains(title, case=False, na=False)]
        
        if results.empty:
            return None, None, "No papers found with that title."
        
        pid = results.iloc[0]["paper_id"]
        pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"
        
        with st.spinner(f"Fetching paper {pid}..."):
            response = requests.get(pdf_url)
            pdf_file = BytesIO(response.content)
            
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())
            
            loader = UnstructuredPDFLoader("temp.pdf", strategy="hi_res")
            docs = loader.load()
            
            clean_docs = []
            for doc in docs:
                clean_text = strip_references_regex(doc.page_content)
                clean_docs.append(Document(page_content=clean_text, metadata=doc.metadata))
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150,
                separators=["\n\n", ".\n", ";", "\n", ". ", " "]
            )
            chunks = text_splitter.split_documents(clean_docs)
            
            # --- VectorDB (in memory, no cache) ---
            embeddings = get_embeddings()
            vectordb = Chroma.from_documents(
                chunks,
                embedding=embeddings
            )
            
            os.remove("temp.pdf")
            
            # Setup chain
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=gemini_api_key,
                temperature=0.3
            )
            
            base_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            mmr_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            
            retriever = EnsembleRetriever(
                retrievers=[base_retriever, mmr_retriever],
                weights=[0.5, 0.5]
            )
            
            parser = StrOutputParser()
            prompt = PromptTemplate(
                template="""You are a helpful assistant.
You remember the previous conversation:
{chat_history}

Try to answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
Try to give output in bullet point or structured way.

{context}
Question: {question}
""",
                input_variables=['chat_history', 'context', 'question']
            )
            
            memory_lambda = RunnableLambda(
                lambda _: st.session_state.memory.load_memory_variables({})["chat_history"]
            )
            
            parallel_chain = RunnableParallel({
                "chat_history": memory_lambda,
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            
            chain = parallel_chain | prompt | model | parser
            
            return vectordb, chain, None
            
    except Exception as e:
        return None, None, str(e)


# UI Layout
st.title("📚 Research Paper Reader")
st.markdown("Ask questions about research papers from arXiv")

# Sidebar
with st.sidebar:
    st.header("📄 Load Paper")
    paper_title = st.text_input("Enter paper title:", placeholder="e.g., attention is all you need")
    
    if st.button("Load Paper", type="primary"):
        if paper_title:
            with st.spinner("Loading paper..."):
                vectordb, chain, error = load_paper(paper_title)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.vectordb = vectordb
                    st.session_state.chain = chain
                    st.session_state.paper_loaded = True
                    st.session_state.messages = []
                    st.success("✅ Paper loaded successfully!")
        else:
            st.warning("Please enter a paper title")
    
    if st.session_state.paper_loaded:
        st.success("✅ Paper is loaded and ready!")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=2
            )
            st.rerun()

# Main chat area
if not st.session_state.paper_loaded:
    st.info("👈 Please load a paper from the sidebar to start chatting")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the paper..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke(prompt)
                    st.markdown(response)
                    
                    # Save to memory
                    st.session_state.memory.save_context(
                        {"input": prompt}, 
                        {"output": response}
                    )
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
