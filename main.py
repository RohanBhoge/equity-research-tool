import os
import streamlit as st
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
# Document Loading and Splitting
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store and Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# LLM and Chain Components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
# Make sure you have a .env file with your GOOGLE_API_KEY
load_dotenv()

# --- App Configuration ---
# Define the path for the persistent Chroma database
persist_directory = 'db_gemini_chroma'

# --- Streamlit App UI ---
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# --- Main Logic ---

# 1. When the user clicks "Process URLs"
if process_url_clicked:
    # Filter out empty URLs
    urls = [url for url in urls if url.strip()]
    if not urls:
        main_placeholder.error("Please enter at least one valid URL.")
    else:
        try:
            # Load data from URLs
            main_placeholder.text("Data Loading...Started...✅")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Split data into chunks
            main_placeholder.text("Text Splitting...Started...✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(data)

            # Create embeddings using Gemini
            main_placeholder.text("Creating Embeddings...Started...✅")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))

            # Create and persist the Chroma vector store
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            main_placeholder.success("URLs processed and vector store created successfully! ✅")

        except Exception as e:
            main_placeholder.error(f"An error occurred: {e}")


# 2. For handling the user's question
query = st.text_input("Question: ")
if query:
    # Check if the vector store has been created
    if os.path.exists(persist_directory):
        try:
            # Load the persisted vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever()

            # Define the LLM and the LCEL Chain
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,google_api_key=os.getenv("GEMINI_API_KEY"))

            template = """
            Answer the question based only on the following context.
            Cite the source from the metadata if available.

            Context:
            {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'N/A')}" for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Invoke the chain and display the result
            result = chain.invoke(query)
            st.header("Answer")
            st.write(result)

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
    else:
        st.warning("Please process URLs first before asking a question.")