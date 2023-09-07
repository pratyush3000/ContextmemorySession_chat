from dotenv import load_dotenv
import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain.chains.question_answering import load_qa_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)  # read and manipulates pdf file
        text = ""         # initialise empty string variable

        for page in pdf_reader.pages:  # loop to iterate pages
            text += page.extract_text()  # extract text and appends it to text variable

        text_splitter = RecursiveCharacterTextSplitter(  # creates a recursivetextsplitter object called txtsplitter

            chunk_size=100,
            chunk_overlap=20,
            length_function=len,

        )

        # split extracted text into chunks
        chunks = text_splitter.split_text(text)

        # initialise object called embeddings from this class to generate word embeddings
        embeddings = OpenAIEmbeddings()
        # create a vector store named vectorstore using faiss library
        Vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        # extract first 4 character from pdf and assign it to variable store_name
        store_name = pdf.name[:4]
        if os.path.exists(f"{store_name}.pkl"):
          # this condition checks if a pickle fil with the same name as store_name and .pkl file exist
            with open(f"{store_name}.pkl", "rb") as f:
                pickle.load(f)
              # if the pickle file xist it opens fil in binary read mode and load its content using pickle load but the loaded data is not assigned to any variable,so it doesn't have any effect
        else:
            # if pickle file doesnt exist it enters this branch
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(Vectorstore, f)

        llm = OpenAI(temperature=0)  # can affect randomness
        history = MongoDBChatMessageHistory(
            connection_string="mongodb://localhost:27017", session_id="user_session-1"
        )

        memory = ConversationBufferMemory(  # ConversationBufferMemory is a type of memory in Langchain that keeps a buffer of the recent interactions in a conversation.
            memory_key='chat_history', return_messages=True, output_key='answer', chat_memory=history)

        retriever = Vectorstore.as_retriever()

        qachat = ConversationalRetrievalChain.from_llm(  # ConversationalRetrievalChain is a chain in LangChain designed for conversational question answering over documents. It allows you to ask followup questions, taking into account previous questions and answers in the conversation.
            # it provides an easy way to add conversational QA abilities to an existing document storage/retrieval system.
            llm=llm,
            memory=memory,
            retriever=retriever

        )
        count = 0
        while(True):
            query = st.text_input("you: ", key=count)
            if query.lower() == "exit":
                break

            response = qachat({"question": query})
            res1 = response["answer"]
            st.write(res1)
            count = count+1


if __name__ == "__main__":
    main()
