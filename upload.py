import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 파일 업로드
uploaded_files = st.file_uploader("PDF파일들을 선택해 주세요(Image PDF는 안되요).", type=['pdf'], accept_multiple_files=True) 

# PDF를 문서로 변환하는 함수
def pdfs_to_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        documents.extend(pages)
    return documents

# 업로드된 PDF 파일 처리
if uploaded_files:
    pages = pdfs_to_documents(uploaded_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20, length_function=len, is_separator_regex=False)
    texts = text_splitter.split_documents(pages)
    embeddings_model = OpenAIEmbeddings()

    # 저장할 디렉토리 설정
    persist_directory = "/chroma"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    db = Chroma.from_documents(texts, embeddings_model, persist_directory=persist_directory)
