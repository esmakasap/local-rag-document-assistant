from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("1. PDF okunuyor...")
loader = PyPDFLoader("Mikroişlemciler.pdf")
docs = loader.load()

print("2. Metin parçalanıyor...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
chunks = text_splitter.split_documents(docs)

print("3. Vektör veritabanı oluşturuluyor...")
embeddings = FastEmbedEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("4. Llama 3.2 modeli bağlanıyor...")
llm = Ollama(model="llama3.2", temperature=0.2)

template = """Aşağıda bir ders belgesinden alınan metin parçaları verilmiştir. 
Bu metne dayanarak sorulan soruya son derece akıcı, kurallı ve profesyonel bir Türkçe ile yanıt ver.

Belge Metni:
{context}

Soru: {question}

Cevap (Sadece belgedeki bilgiyi özetle):"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n--- SİSTEM HAZIR! Soru sorabilirsin (Çıkmak için 'q' yaz) ---\n")

while True:
    soru = input("\nSorun: ")
    if soru.strip().lower() == 'q':
        break
    if not soru.strip():
        continue
    
    print("\nCevap üretiliyor...\n")
    cevap = rag_chain.invoke(soru)
    print(f"Cevap:\n{cevap.strip()}\n")
    print("-" * 50)