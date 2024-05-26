import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("AskArxiv 📖")
st.text("This tool will help you answer any question related to a research paper from Arxiv!")

placeholder = st.empty()
ip = placeholder.text_input("Enter research paper identifier: ")

llm = OpenAI(temperature=0.9, max_tokens=100)

if ip:
    placeholder.text("Loading...")
    # load the research paper
    link = 'https://arxiv.org/pdf/'+ip
    loader = PyPDFLoader(link)
    paper = loader.load()

    # create chunks of data
    paper_text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    chunks = paper_text_splitter.split_documents(paper)

    # embeddings for chunks and add them to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorDB = FAISS.from_documents(chunks, embeddings)

    # getting 5 most similar chunks and using them to find the answer to the user query
    retriever = vectorDB.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True
    )
    placeholder.text("")
    query_placeholder = st.empty()
    query = query_placeholder.text_input("What do you want to know about this paper?")
    if query:
        query_placeholder.text("Loading...")
        result = chain({'query': query})
        query_placeholder.text("")
        st.write(result['result'])
