import csv
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.chains import RetrievalQA

load_dotenv()

csv.field_size_limit(sys.maxsize)

# i loaded the data
loader = CSVLoader(file_path="arunachal_tourism_final_cleaned.csv")
data = loader.load()
print(data[0])

# text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_documents(data)
print(f"Total chunks created: {len(texts)}")

# storing in vectordb

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory="./arunachal_db"
)

# Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# llm
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=500,
    temperature=0.7,
)
chat_llm = ChatHuggingFace(llm=llm)
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_llm, chain_type="stuff", retriever=retriever, return_source_documents=False
)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Thankyou so much!!")
        break

    if user_input.strip() == "":
        continue

    response = qa_chain.invoke(user_input)

    print(f"\nBot: {response['result']}")
