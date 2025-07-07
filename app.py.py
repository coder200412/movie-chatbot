


import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

st.set_page_config(page_title="Movie Chatbot", layout="wide")
st.title("🎬 Movie Metadata Chatbot (RAG-based)")

uploaded_file = st.file_uploader("Upload your movies_metadata.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")

    rows = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    documents = [Document(page_content=row) for row in rows]

   
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = splitter.split_documents(documents)
    docs = docs[:500]  # Optional limit
    st.info(f"Vectorizing {len(docs)} chunks...")

   
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectorstore = FAISS.from_documents(docs, embeddings_model)

  
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})


    prompt_template = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""Use the following summaries to answer the question.
If you don't know, say \"I don't know\".

{summaries}

Question: {question}
Answer:"""
    )


    chatpipe = pipeline("text2text-generation", model="google/flan-t5-base", device=-1, max_length=256)
    llm = HuggingFacePipeline(pipeline=chatpipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={"combine_prompt": prompt_template}
    )

  
    st.subheader("Ask about any movie in the dataset")
    user_query = st.text_input("Your question:", placeholder="e.g. Who directed Titanic?")

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(user_query)
            answer = response["result"] if isinstance(response, dict) else response
            st.success("Answer: " + answer)


st.sidebar.title("🚀 Deployment Tips")
st.sidebar.markdown("""
**To deploy this app:**

1. **Install dependencies**
```bash
pip install streamlit langchain langchain-community langchain-huggingface transformers pandas
```

2. **Save this file as `app.py`**

3. **Run locally**
```bash
streamlit run app.py
```

4. **Deploy on [Streamlit Cloud](https://streamlit.io/cloud)**:
    - Push this file and your `movies_metadata.csv` to a public GitHub repo
    - Connect your repo to Streamlit Cloud
    - Add these Python packages to `requirements.txt`

    ```
    streamlit
    pandas
    langchain
    langchain-community
    langchain-huggingface
    transformers
    ```
""")
