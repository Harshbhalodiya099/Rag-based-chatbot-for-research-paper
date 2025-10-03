import os 
import re
import requests
import time
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

from langchain.schema import Document 
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

# 2. Retrieve the GEMINI_API_KEY from the environment
gemini_api_key = os.getenv("GEMINI_API_KEY")


df = pd.read_csv("papers_title.csv")

title = input("write title of paper you want to fetch: ")  # like "attention is all you need"

def search_paper(title_query: str, top_k: int = 5):
    results = df[df['title'].str.contains(title_query, case=False, na=False)]
    return results.head(top_k)


def get_paper_by_id(paper_id: str):
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    return [Document(page_content="", metadata={"pdf_url": pdf_url})]


# ---------------- USAGE ---------------- #

# Search
results = search_paper(title)
print("🔎 Search Results:")
print(results)

# Fetch
pdf_url = None

if not results.empty:
    pid = results.iloc[0]["paper_id"]
    print(f"\n📄 Fetching paper {pid} ...")
    docs = get_paper_by_id(pid)

    for doc in docs:
        pdf_url = doc.metadata.get("pdf_url")
        print("PDF URL:", pdf_url)


# Load PDF directly from URL

# Fetch PDF from URL
response = requests.get(pdf_url)
pdf_file = BytesIO(response.content)

# Save temporarily
with open("temp.pdf", "wb") as f:
    f.write(pdf_file.getbuffer())

# Load using unstructured loader
loader = UnstructuredPDFLoader("temp.pdf",strategy="hi_res")    #,mode="elements". 
docs = loader.load()

def strip_references_regex(text):
    # Split at "References" first
    split_text = re.split(r"\nReferences\n|\nREFERENCES\n|\nBibliography\n", text, flags=re.IGNORECASE)
    main_text = split_text[0]

    # Optionally remove lines that look like references at the end
    lines = main_text.split("\n")
    filtered_lines = []
    for line in lines:
        if re.match(r"\[\d+\]", line.strip()):  # lines starting with [1], [2], etc
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)

clean_docs = []
for doc in docs:
    clean_text = strip_references_regex(doc.page_content)
    clean_docs.append(Document(page_content=clean_text, metadata=doc.metadata))



# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,
    chunk_overlap=150,
    separators=["\n\n", ".\n", ";", "\n", ". ", " "])
chunks = text_splitter.split_documents(clean_docs)

print(len([chunk.page_content for chunk in chunks]))



# Store in vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_api_key)

# Batch process with delays
def create_vectordb_with_batching(chunks, embeddings, batch_size=20):
    """Process chunks in batches to avoid rate limits"""
    all_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        if i > 0:
            time.sleep(10)  # Wait 60 seconds between batches
        
        if i == 0:
            vectordb = Chroma.from_documents(
                batch, 
                embedding=embeddings, 
                persist_directory="vectorstore"
            )
        else:
            vectordb.add_documents(batch)
    
    return vectordb

# Use it:
vectordb = create_vectordb_with_batching(chunks, embeddings, batch_size=25)

# vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="vectorstore")


print(f"✅ Stored full paper in vector store! Total chunks: {len(chunks)}")



# Remove temporary pdf file
os.remove("temp.pdf")

# Load the persisted vector store
loaded_vdb = Chroma(
persist_directory="./vectorstore",
embedding_function=embeddings)





model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=gemini_api_key,temperature=0.3)

# Form a retriever
base_retriever = loaded_vdb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
mmr_retriever = loaded_vdb.as_retriever(search_type="mmr", search_kwargs={"k": 5})  # Using MMR as retriever

retriever = EnsembleRetriever(
    retrievers=[base_retriever, mmr_retriever],
    weights=[0.5, 0.5]
)

# form  String output parser
parser = StrOutputParser()

# # Build prompt template
# prompt = PromptTemplate(
#         template="""You are a helpful assistant.
#         Try to answer ONLY from the provided transcript context.
# If the context is insufficient, just say you don't know.
# try to give give output in bullet point or structured way.
# {context}
# Question: {question}
# """,
# input_variables = ['context','question']
# )

# for join returned document 
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


# Keep only the last 2 exchanges (user + agent)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2   # <-- keep only last 2 interactions
)
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
    input_variables=['chat_history','context','question']
)

# Lambda to inject only last 2 interactions
memory_lambda = RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"])

parallel_chain = RunnableParallel(
    {
        "chat_history": memory_lambda,
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)

chain = parallel_chain | prompt | model | parser

while True:
    user_input = input("\nEnter your question (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    
    # Run the chain
    response = chain.invoke(user_input)
    print("\nAssistant:", response)
    
    # Save to memory
    memory.save_context({"input": user_input}, {"output": response})





