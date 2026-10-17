import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os
import re
import html

st.set_page_config(page_title="PDF Search With Citations", layout="centered")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero h1 {font-size: 2.2rem; margin-bottom: 0;}
.hero p {color: #9fb0c8; margin-top: 4px;}
.card {
    background: #131c2e;
    border: 1px solid #1f2d46;
    border-left: 4px solid #2dd4bf;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.badge {
    background: #0f766e;
    color: #ffffff;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    margin-left: 10px;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_model():
    return FastEmbedEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


def remove_references(docs):
    cleaned = []
    for d in docs:
        parts = re.split(r"\n\s*References\s*\n", d.page_content)
        if len(parts) > 1:
            d.page_content = parts[0]
            cleaned.append(d)
            break
        cleaned.append(d)
    return cleaned

def build_index(uploaded_files, model):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name

        if ext == ".pdf":
            docs = remove_references(PyPDFLoader(path).load())
        elif ext == ".docx":
            docs = Docx2txtLoader(path).load()
        else:
            docs = TextLoader(path, encoding="utf-8").load()

        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks, start=1):
            if ext == ".pdf":
                c.metadata["label"] = f"{f.name} - page {c.metadata['page'] + 1}"
            else:
                c.metadata["label"] = f"{f.name} - part {i}"
        all_chunks.extend(chunks)

    return FAISS.from_documents(all_chunks, model)

# retriver process
def get_answer(question,vector_store,model):
    retriver=vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})
    retriver_docs=retriver.invoke(question)

    sentence = []
    pages = []
    for splits in retriver_docs:
        text = splits.page_content.replace("\n", " ")
        sentence_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        for s in sentence_splits:
            f = s.strip()
            if len(f) > 30:
              sentence.append(f)
              pages.append(splits.metadata['label'])
    if len(sentence) == 0:
        return [], retriver_docs

    question_vector=np.array(model.embed_query(question))
    sentences_vector=np.array(model.embed_documents(sentence))

    score=sentences_vector @ question_vector/(np.linalg.norm(sentences_vector,axis=1)*np.linalg.norm(question_vector))

    top_indx=np.argsort(score)[::-1][:3]

    result=[]
    for i in top_indx:
        result.append((sentence[i],pages[i]))
    return result,retriver_docs
    
# now write the UI code in streamlit

st.markdown(
    "<div class='hero'><h1>PDF Search With Citations</h1>"
    "<p>Upload a PDF, ask a question, get the matching lines with page numbers.</p></div>",
    unsafe_allow_html=True,
)

model = build_model()

with st.sidebar:
    st.header("Your document")
    files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process files", use_container_width=True) and files:
        with st.spinner("Indexing..."):
            st.session_state['db'] = build_index(files, model)
            st.session_state['name'] = ", ".join([f.name for f in files])
        st.success("Files are ready")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What PSNR and SSIM does the proposed method achieve?",
)

if 'db' not in st.session_state:
    st.info("Upload a PDF from the sidebar and click Process PDF to start.")

if question and 'db' in st.session_state:
    results, hits = get_answer(question, st.session_state['db'], model)

    for sent, label in results:
        st.markdown(
            f"<div class='card'>{html.escape(sent)}<span class='badge'>{html.escape(label)}</span></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Sources")
    for h in hits:
        with st.expander(h.metadata['label']):
            st.write(h.page_content)

st.markdown(
    "<hr><p style='text-align:center; color:#9fb0c8;'>Made by <b>Fawad Ahmad Bilal</b></p>",
    unsafe_allow_html=True,
)




# for single ducoment loader
# def build_index(uploaded_file,model):
#     with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp:
#         tmp.write(uploaded_file.read())
#         path=tmp.name
# # load the documnets in the docs variables
#     docs=PyPDFLoader(path).load()
#     docs=remove_references(docs)
# # splitting the text
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#     chunk=text_splitter.split_documents(docs)
#     # embed the text and then save it in the vector store
#     return FAISS.from_documents(chunk,model)