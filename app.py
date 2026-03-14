import os
import sys
import csv
import streamlit as st
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Arunachal AI Guide", page_icon="🏔️", layout="wide")


# ---------------------------------------------------
# BRIGHTER CINEMATIC CSS
# ---------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

/* Remove all spacing */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    font-family: 'Poppins', sans-serif;
    color: white;
}

/* Background image everywhere */
[data-testid="stAppViewContainer"],
[data-testid="stBottomBlockContainer"] {
    background: url("https://images.unsplash.com/photo-1501785888041-af3ef285b470?q=80&w=3000&auto=format&fit=crop")
                no-repeat center center fixed !important;
    background-size: cover !important;
    position: relative;
}

/* 🔻 Darken brightness slightly */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.28);  /* increase for darker, decrease for lighter */
    z-index: 0;
}

/* Keep content above overlay */
.block-container {
    position: relative;
    z-index: 1;
    padding: 6rem 4rem 0rem 4rem !important;
    max-width: 100% !important;
}

/* Remove footer/header gap */
footer, header {
    visibility: hidden;
}

/* Header styling */
.topbar {
    text-align: center;
    margin-bottom: 30px;
    text-shadow: 0 4px 25px rgba(0,0,0,0.8);
}
.topbar h1 {
    font-size: 52px;
    margin-bottom: 8px;
}
.topbar p {
    font-size: 20px;
    color: #e2e8f0;
}

/* Chat bubbles */
.stChatMessage {
    border-radius: 18px !important;
    padding: 14px !important;
    margin-bottom: 14px !important;
    background: rgba(0,0,0,0.55) !important;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(8px);
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(0,0,0,0.45) !important;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.45);
}
</style>
""",
    unsafe_allow_html=True,
)

# st.markdown(
#     """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

# /* Remove all spacing */
# html, body, .stApp, [data-testid="stAppViewContainer"] {
#     margin: 0 !important;
#     padding: 0 !important;
#     height: 100% !important;
#     font-family: 'Poppins', sans-serif;
#     color: white;
# }

# /* Background image everywhere */
# [data-testid="stAppViewContainer"],
# [data-testid="stBottomBlockContainer"] {
#     background: url("https://images.unsplash.com/photo-1501785888041-af3ef285b470?q=80&w=3000&auto=format&fit=crop")
#                 no-repeat center center fixed !important;
#     background-size: cover !important;
# }

# /* Remove default container width */
# .block-container {
#     padding: 6rem 4rem 0rem 4rem !important;
#     max-width: 100% !important;
# }

# /* Remove footer/header gap */
# footer, header {
#     visibility: hidden;
# }

# /* Header styling */
# .topbar {
#     text-align: center;
#     margin-bottom: 30px;
#     text-shadow: 0 4px 25px rgba(0,0,0,0.8);
# }
# .topbar h1 {
#     font-size: 52px;
#     margin-bottom: 8px;
# }
# .topbar p {
#     font-size: 20px;
#     color: #e2e8f0;
# }

# /* Chat bubbles (same brightness) */
# .stChatMessage {
#     border-radius: 18px !important;
#     padding: 14px !important;
#     margin-bottom: 14px !important;
#     background: rgba(0,0,0,0.55) !important;
#     border: 1px solid rgba(255,255,255,0.2);
#     backdrop-filter: blur(8px);
# }

# /* Chat input transparent so image shows */
# [data-testid="stChatInput"] {
#     background: rgba(0,0,0,0.45) !important;
#     border-radius: 25px;
#     border: 1px solid rgba(255,255,255,0.25);
#     color: white;
# }

# /* Sidebar */
# section[data-testid="stSidebar"] {
#     background: rgba(0,0,0,0.45);
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )


# ---------------------------------------------------
# INITIALIZE BOT
# ---------------------------------------------------
@st.cache_resource
def initialize_bot():
    load_dotenv()
    csv.field_size_limit(sys.maxsize)

    loader = CSVLoader(file_path="arunachal_tourism_final_cleaned.csv")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(data)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory="./arunachal_db",
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=500,
        temperature=0.7,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    chat_llm = ChatHuggingFace(llm=llm)

    return RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=retriever,
    )


qa_chain = initialize_bot()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    """
<div class="topbar">
    <h1>🏔️ Arunachal AI Travel Guide</h1>
    <p>Explore the Land of the Rising Sun with an intelligent companion</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# CHAT MEMORY
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------
# CHAT
# ---------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about Tawang, Ziro, monasteries, festivals..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🎋"):
        with st.spinner("Finding the best travel insights for you..."):
            try:
                response = qa_chain.invoke(user_input)
                bot_answer = response["result"]
                st.write(bot_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_answer}
                )
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## 🌄 About This Guide")
    st.write(
        "This AI assistant is trained on authentic Arunachal Pradesh tourism data to provide accurate travel guidance, hidden gems, and cultural insights."
    )

    st.markdown("---")
    st.markdown("### 🧭 Try asking:")
    st.markdown("""
    - Best places to visit in Tawang  
    - Ziro music festival details  
    - Local food to try  
    - Hidden villages to explore  
    """)
