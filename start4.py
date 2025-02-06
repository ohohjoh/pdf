import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ✅ 페이지 설정을 가장 상단에 배치
st.set_page_config("PDF 문서 분석 챗봇", layout="wide")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re
import uuid  # UUID 추가
from PyPDF2 import PdfMerger

# 환경변수 불러오기
from dotenv import load_dotenv, dotenv_values
load_dotenv()


############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장

from datetime import datetime

def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    # 파일 이름(확장자 제외)을 폴더 이름으로 사용
    base_filename = os.path.splitext(uploadedfile.name)[0]
    temp_dir = f"PDF_임시폴더\\{base_filename}"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 파일 저장
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read())

    return file_path


## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path: str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
        d.metadata['file_name'] = os.path.basename(pdf_path)
        d.metadata['page'] = d.metadata.get('page', 0)
        d.metadata['source'] = pdf_path  # 메타데이터에 source 추가
    documents.extend(doc)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장 (기존 인덱스에 추가)
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")


############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 5개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 5})
    retrieve_docs: List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs


def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()


############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []

    # 파일 이름(확장자 제외)을 폴더 이름으로 사용
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join("PDF_이미지", base_filename)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_path = os.path.join(output_folder, f"{base_filename}_page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths


def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def main():
    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.header("PDF 문서 분석 챗봇")

        pdf_docs = st.file_uploader("PDF Uploader", type="pdf", accept_multiple_files=True)
        button = st.button("PDF 업로드하기")
        if pdf_docs and button:
            all_documents = []
            with st.spinner("PDF문서 저장중"):
                for pdf_doc in pdf_docs:
                    pdf_path = save_uploadedfile(pdf_doc)
                    pdf_document = pdf_to_documents(pdf_path)
                    smaller_documents = chunk_documents(pdf_document)
                    all_documents.extend(smaller_documents)
                    save_to_vector_store(all_documents)

            with st.spinner("PDF 페이지를 이미지로 변환 중..."):
                all_images = []
                base_dir = "PDF_임시폴더"

                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith(".pdf"):
                            pdf_path = os.path.join(root, file)
                            images = convert_pdf_to_images(pdf_path)
                            all_images.extend(images)

                st.session_state.images = all_images

        user_question = st.text_input("PDF 문서에 대해서 질문해주세요", placeholder="무순위 청약 시에도 부부 중복신청이 가능한가요?")

        if user_question:
            response, context = process_question(user_question)
            st.text(response)

        
            for i, document in enumerate(context):  # enumerate로 index 추가
                with st.expander(f"관련 문서 {i+1}"):  # expander label 변경
                    st.text(document.page_content)
                    file_path = document.metadata.get('source', '')
                    page_number = document.metadata.get('page', 0) + 1

                    # key를 index로 변경하여 고유하게 유지
                    if st.button(f"\U0001FAE0 {os.path.basename(file_path)} \n\n pg.{page_number}", key=f"link_{i}"):
                        st.session_state.selected_file = file_path  # 파일 경로 저장
                        st.session_state.selected_page = page_number  # 페이지 번호

                        
    with right_column:
        if st.session_state.get("selected_file") and st.session_state.get("selected_page"):
            file_path = st.session_state.get("selected_file")
            page_number = st.session_state.get("selected_page")

            st.markdown(f'<span style="color:red">선택된 파일:</span> {os.path.basename(file_path)}', unsafe_allow_html=True)
            st.markdown(f'<span style="color:red">선택된 페이지:</span> {page_number}', unsafe_allow_html=True)
            # 이미지 표시
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            image_folder = os.path.join("PDF_이미지", base_filename)
            image_name = f"{base_filename}_page_{page_number}.png"
            image_path = os.path.join(image_folder, image_name)

            if os.path.exists(image_path):
                st.image(image_path, width=600)
            else:
                st.text("이미지를 찾을 수 없습니다.")
        else:
            st.text("선택된 파일/페이지가 없습니다.")

    # with right_column:
    #     page_number = st.session_state.get('page_number')
    #     file_name = st.session_state.get('file_name')
    #     st.text(file_name)

        # st.text(f"페이지 번호: {page_number}")
        # st.text(f"파일 이름: {file_name}")

        # if st.session_state.get('button_clicked') and page_number and file_name:
        #     image_folder = os.path.join("PDF_이미지", os.path.splitext(file_name)[0])

        #     if os.path.exists(image_folder):
        #         images = sorted(os.listdir(image_folder), key=natural_sort_key)
        #         if 0 < page_number <= len(images):
        #             image_path = os.path.join(image_folder, images[page_number - 1])
        #             display_pdf_page(image_path, page_number)
        #         else:
        #             st.warning("페이지가 존재하지 않습니다.")
        #     else:
        #         st.warning("이미지 폴더를 찾을 수 없습니다.")
 
if __name__ == "__main__":
    main()

