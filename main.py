import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# 질문을 받는 부분
st.header("PDF에게 질문해 보세요.")
question = st.text_input('질문을 입력하세요.')

# 질문 답변 처리
if st.button('질문하기'):
    with st.spinner('Wait for it...'):
        # 저장된 DB 로드
        persist_directory = "/chroma"
        db = Chroma.load(persist_directory)

        # LLM 설정
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # 질문과 답변 처리
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        result = qa_chain({"query": question})
        st.write(result["result"])
