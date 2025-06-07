import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
# IMPORTANT: Make sure this matches the name of your uploaded PDF file!
PDF_FILE_PATH = "3.pdf"

# Attempt to get the API key from Replit Secrets
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- SYSTEM PROMPT DEFINITION ---
# This is where you define HOW your AI should behave and its persona.
# The {context} and {question} are placeholders that LangChain will fill.
SYSTEM_PROMPT_TEMPLATE = """
You are a highly specialized AI assistant. Your ONLY purpose is to answer questions accurately based on the content of the PDF document provided to you.
If the information needed to answer the question is NOT found within the document's provided context, you MUST clearly state: "I'm sorry, but I cannot find that specific information within the provided document."
Do NOT make up information, hallucinate, or answer questions outside the scope of the document. Be polite, concise, and strictly stick to the document's content.

Use the following pieces of context to answer the question at the end.
Context: {context}
Question: {question}
Helpful Answer:"""

# --- HELPER FUNCTION TO LOAD AND PROCESS PDF (CACHED FOR EFFICIENCY) ---
# Streamlit's @st.cache_resource decorator ensures this function runs only once
# unless the PDF_FILE_PATH changes, saving processing time.
@st.cache_resource
def initialize_pdf_processor_and_qa_chain(pdf_path, api_key):
    if not api_key:
        st.error("Google API Key not found. Please set it in Replit Secrets as GOOGLE_API_KEY.")
        return None

    if not os.path.exists(pdf_path):
        st.error(f"Error: The PDF file '{pdf_path}' was not found. Please ensure it's uploaded to Replit and the name is correct in the script.")
        return None

    st.info(f"Processing PDF: {pdf_path}...")
    try:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            st.error("Could not load any content from the PDF. It might be empty or corrupted.")
            return None
        st.info(f"Successfully loaded {len(documents)} pages from the PDF.")

        # 2. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        if not docs:
            st.error("Could not split the document into text chunks. The PDF might not contain extractable text.")
            return None
        st.info(f"PDF split into {len(docs)} text chunks.")

        # 3. Create Embeddings (using Gemini)
        st.info("Generating text embeddings with Google Gemini...")
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # A common Gemini embedding model
            google_api_key=api_key
        )

        # 4. Create Vector Store (FAISS)
        st.info("Creating vector store for similarity search...")
        vector_store = FAISS.from_documents(docs, embeddings_model)
        st.success("PDF processed and vector store created successfully!")

        # 5. Initialize LLM (Gemini Pro)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",  # Or "gemini-1.0-pro", "gemini-1.5-pro-latest" etc.
            google_api_key=api_key,
            temperature=0.1,  # Lower temperature for more factual, less creative answers
            convert_system_message_to_human=True # Often helpful for Gemini with LangChain
        )

        # 6. Create Prompt Template
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=SYSTEM_PROMPT_TEMPLATE,
        )

        # 7. Create RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" puts all relevant chunks into the prompt
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant chunks
            return_source_documents=True,  # So we can see which parts of PDF were used
            chain_type_kwargs={"prompt": qa_prompt} # THIS IS HOW WE USE OUR CUSTOM SYSTEM PROMPT
        )
        return qa_chain

    except Exception as e:
        st.error(f"An error occurred during PDF processing or QA chain initialization: {e}")
        return None

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Gemini PDF Q&A Bot", layout="wide")
st.title("📄 Chat with Your PDF using Google Gemini")
st.markdown("Ask questions about the PDF document you've uploaded. The AI will answer based *only* on its content.")

# Check for API Key first
if not GOOGLE_API_KEY:
    st.warning("🔴 Google API Key is not configured. Please add `GOOGLE_API_KEY` to your Replit Secrets.")
else:
    # Initialize the QA chain (this will also process the PDF if not already cached)
    qa_chain_instance = initialize_pdf_processor_and_qa_chain(PDF_FILE_PATH, GOOGLE_API_KEY)

    if qa_chain_instance:
        st.sidebar.header("Ask a Question:")
        user_question = st.sidebar.text_input("Enter your question about the PDF:", key="user_question_input")

        if user_question:
            with st.spinner("Gemini is thinking and searching the document..."):
                try:
                    result = qa_chain_instance({"query": user_question})
                    st.subheader("💡 Gemini's Answer:")
                    st.write(result["result"]) # This is where the AI's answer will appear

                    # Optionally, display the source document chunks used for the answer
                    with st.expander("📚 Show Source Document Chunks Used by Gemini"):
                        if result.get("source_documents"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"--- **Chunk {i+1} (from page {doc.metadata.get('page', 'N/A')})** ---")
                                st.caption(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        else:
                            st.write("No source documents were explicitly returned.")
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")
    else:
        st.info("The Q&A system could not be initialized. Please check error messages above.")

st.sidebar.markdown("---")
st.sidebar.info(f"Ensure your PDF is uploaded as `{PDF_FILE_PATH}` (or update the variable in `main.py`) and your `GOOGLE_API_KEY` is correctly set in Replit Secrets.")