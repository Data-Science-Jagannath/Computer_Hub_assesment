import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.llms import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
# from langchain_community.document_loaders import PyPDFLoader


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    count=0
    for page in pdf_reader.pages:
        text += page.extract_text()
        count += 1
    
    return text, count


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,)

    chunks = text_splitter.split_text(text)

    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def ll_retriver(vectorstore):
    llm = OpenAI(temperature=0)
    # llm = Cohere(temperature=0)
    llm_based_retriver=MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )
    return llm_based_retriver

def chain(llm_based_retriever):
    llm = OpenAI(temperature=0)
    # llm = Cohere(temperature=0)
    memory = ConversationBufferWindowMemory(size=2) # remember only last 2 conversations
    QA_Chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=llm_based_retriever,
        memory=memory
    )
    return QA_Chain

pdf_path = r"C:\Users\30jag\OneDrive\Desktop\Resolute_AI\Corpus.pdf"

def main():
    load_dotenv()

    if 'raw_text' not in st.session_state:
        raw_text, _ = get_pdf_text(pdf_path)
        st.session_state['raw_text'] = raw_text

    # Generate and cache text chunks in session state
    if 'text_chunks' not in st.session_state:
        text_chunks = get_text_chunks(st.session_state['raw_text'])
        st.session_state['text_chunks'] = text_chunks

    # Create and cache vector store in session state
    if 'vectorstore' not in st.session_state:
        vectorstore = get_vectorstore(st.session_state['text_chunks'])
        st.session_state['vectorstore'] = vectorstore

    # Create and cache LLM-based retriever in session state
    if 'llm_based_retriever' not in st.session_state:
        llm_based_retriever = ll_retriver(st.session_state['vectorstore'])
        st.session_state['llm_based_retriever'] = llm_based_retriever

    # Create and cache QA chain in session state
    if 'QA_Chain' not in st.session_state:
        QA_Chain = chain(st.session_state['llm_based_retriever'])
        st.session_state['QA_Chain'] = QA_Chain

    st.title(":blue[Document Q&A with LangChain] :books:",)
    
    st.write(":red[This chatbot is trained on a single internal document.]")

    question = st.text_input("Ask a Question about your document:")
    if question:
        with st.spinner("Getting answer..."):
            try:
                response = st.session_state['QA_Chain']({"query": question})
                # "chat_history": st.session_state.get('chat_history', [])
                st.write(response['result']) 
                st.session_state['chat_history'] = response.get('chat_history', [])
            except Exception as e:
                st.error(f"Error getting answer: {e}")

if __name__ == "__main__":
    main()