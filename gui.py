import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local RAG Asistanı", layout="centered")
st.title("📄 Local RAG - Doküman Asistanı")
st.caption("Yerel LLM (Llama 3.2) ve ChromaDB ile Doküman Analizi")

DB_DIR = "chroma_db"

@st.cache_resource(show_spinner="Doküman indeksleniyor, lütfen bekleyin...")
def get_rag_chain():
    embeddings = FastEmbedEmbeddings()
    
    # Daha önce veritabanı kaydedilmişse tekrar okuma yapmadan oradan yükle
    if os.path.exists(DB_DIR):
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        loader = PyPDFLoader("Mikroişlemciler.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        chunks = text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama3.2", temperature=0.2)
    
    template = """Aşağıdaki belge metnini kullanarak sorulan soruya akıcı ve profesyonel bir Türkçe ile yanıt ver.
Belge Metni:
{context}

Soru: {question}
Cevap:"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = get_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("PDF hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt üretiliyor..."):
            response = rag_chain.invoke(prompt_text)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})