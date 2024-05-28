import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

def get_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def make_vectorstore(chunks, OPENAI_API_KEY):
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)
    return vectorstore

def run_chain(vectorstore, query, history, OPENAI_API_KEY):
    model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model = 'gpt-4-turbo')
    parser = StrOutputParser()

    template = """
    Your are an assistant within a tool that is used to analyze provided PFD documents.
    The document you are about to analyze has been processed into chunks and stored in the local vectorstore. 
    You will be provided with relative context through vector retrieval. 
    You are also provided with history of conversation with user.

    Answer the question based ONLY on the context and history that is provided below. If you can't answer the question, reply "I don't know".

    Context: {context}

    History: {history}

    Question: {question}

    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model | parser

    retrieved_context = ""
    
    docs = vectorstore.similarity_search(query)

    for doc in docs:
        retrieved_context = retrieved_context + "\n" + doc.page_content
    
    response = chain.invoke({
        "context": retrieved_context,
        "question": query,
        "history": history
    })


    return response


def handle_user_request(user_question, history):
    response = run_chain(st.session_state.conversation, st.session_state.vectorstore, user_question, history )
    #response = st.session_state.conversation.invoke(user_question, history)
    st.write(response)

def get_history(messages):
    history = ""
    for message in messages:
        # Append the role and content to the formatted string, followed by a newline for separation
        history += f"{message['role'].capitalize()}: {message['content']}\n"
    return history


load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

st.set_page_config(page_title = "Chat with PDF", page_icon = ":books:")
st.header("Chat with PDF - the simple version")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Please upload, process and then ask question about your PDF")

history = " "

#main window logic
if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({"role":"user", "content":query})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        full_response = ""
        full_response = run_chain(st.session_state.vectorstore, query, get_history(st.session_state.messages), OPENAI_API_KEY)

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# sidebar logic
with st.sidebar:
    st.subheader("Your document")
    pdf_doc = st.file_uploader("Upload your PDF here and click 'Process'", type='pdf')
    if st.button("Process"):
        with st.spinner("Processing..."):
            # get PDF text
            text = get_text(pdf_doc)

            # get chunks
            chunks = get_chunks(text)

            # get vectorstore
            st.session_state.vectorstore = make_vectorstore(chunks, OPENAI_API_KEY)

            # notify the document has been processed
            st.markdown("Your document has been processed. What do you want to know about it? Use the chat window to your right to ask questions.")
