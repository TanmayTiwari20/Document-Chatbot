import os
from apikey import apikey

import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = apikey


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]

st.title("Ask the Doc!")
uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"],accept_multiple_files=True)
add_file = st.button("Add File", on_click=clear_history)

if uploaded_file and add_file:
    for files in uploaded_file:
        with st.spinner('Reading... Chunking... and Embedding the file... Thoda wait karo please :)'):
            bytes_data = files.read()
            file_name = os.path.join("uploaded_files", files.name)
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            name,extension = os.path.splitext(file_name)
            if extension == '.pdf':
                from langchain.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_name)
            elif extension == '.docx':
                from langchain.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_name)
            elif extension == '.txt':
                from langchain.document_loaders import TextLoader
                loader = TextLoader(file_name)
            else:
                st.write('Document format not supported')
            
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents=documents)

            embeddings = OpenAIEmbeddings()
            vector_store = Chroma.from_documents(chunks, embedding=embeddings)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

            retriever = vector_store.as_retriever()
            # chain = RetrievalQA.from_chain_type(llm,retriever=retriever)
            crc = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
            st.session_state.crc = crc
            st.success('File Uploaded, chunked, embedded successfully!')
    
question = st.text_input("What's your question?!")

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        # response = chain.run(question)
        if "history" not in st.session_state:
            st.session_state["history"] = []
        response = crc.run(
            {"question": question, "chat_history": st.session_state["history"]}
        )

        st.session_state["history"].append((question, response))
        st.write(response)
        # for prompts in st.session_state["history"]:
        #     st.write("Question: " + prompts[0])
        #     st.write("Answer: " + prompts[1])
