import streamlit as st
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import os
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

# Embeddings (LOCAL – NO QUOTA)
from langchain.embeddings import HuggingFaceEmbeddings

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- UTILITY FUNCTIONS ---------------- #

# Step 1: Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# Step 2: Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# Step 3: Build FAISS vector store (ONCE)
def build_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# Step 4: Load FAISS safely
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


from langchain_core.runnables import RunnablePassthrough

def get_rag_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(
         model="gemini-3-flash-preview",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using the provided context.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """)

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain





# Step 7: Handle user query
def user_input(question, api_key, history):
    if "vector_store" not in st.session_state:
        st.warning("Please upload and process PDFs first.")
        return

    retriever = st.session_state.vector_store.as_retriever()
    rag_chain = get_rag_chain(api_key, retriever)

    answer = rag_chain.invoke(question).content


    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    intro = rephrase_with_gemini(question, api_key)

    history.append((question, answer, "Gemini", timestamp))

    st.markdown(
        display_chat(intro, answer, timestamp),
        unsafe_allow_html=True
    )


def display_chat(user_msg, bot_msg, timestamp):
    return f"""
    <div style="margin-bottom: 1.5rem;">
        <strong>AI Response</strong>
        <span style="font-size:0.85rem;color:gray;margin-left:10px;">{timestamp}</span>
        <div style="margin-top:6px;line-height:1.6;">
            {user_msg}<br><br>{bot_msg}
        </div>
    </div>
    """


# ---------------- STREAMLIT UI ---------------- #

def main():
    st.set_page_config(
        page_title="Smart Document Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.header("📚 Ask, Learn, Discover – From Your PDFs")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    with st.sidebar:
        st.title("Controls")

        api_key = st.text_input("Google API Key", type="password")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Submit & Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)

                    vector_store = build_vector_store(chunks)
                    st.session_state.vector_store = vector_store

                    st.success("PDFs processed successfully ✅")
            else:
                st.warning("Upload PDFs first.")

        if st.button("Reset Chat"):
            st.session_state.conversation_history = []

        st.markdown("---")
        st.markdown("**Made with ❤️ by Krishna Vig**")

    question = st.text_input("Ask a question from your PDFs")

    if question and api_key:
        user_input(
            question,
            api_key,
            st.session_state.conversation_history
        )

    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">📥 Download Chat History</a>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
