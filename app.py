from dotenv import load_dotenv
import os


import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import MongoDBChatMessageHistory


def main():
    load_dotenv()  # load environment variable from .env
    # open pdf file in binary read mode and assigns it t̥o the variable pdf
    pdf = open("DP1Merrill_Manual_en.pdf", "rb")
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
        history1 = MongoDBChatMessageHistory(
            connection_string="mongodb://localhost:27017", session_id="user_session-1"
        )
        history2 = MongoDBChatMessageHistory(
            connection_string="mongodb://localhost:27017", session_id="user_session-2"
        )

        memory1 = ConversationBufferMemory(  # ConversationBufferMemory is a type of memory in Langchain that keeps a buffer of the recent interactions in a conversation.
            memory_key='chat_history', return_messages=True, output_key='answer', chat_memory=history1)

        memory2 = ConversationBufferMemory(  # ConversationBufferMemory is a type of memory in Langchain that keeps a buffer of the recent interactions in a conversation.
            memory_key='chat_history', return_messages=True, output_key='answer', chat_memory=history2)
        retriever = Vectorstore.as_retriever()

        qachat1 = ConversationalRetrievalChain.from_llm(  # ConversationalRetrievalChain is a chain in LangChain designed for conversational question answering over documents. It allows you to ask followup questions, taking into account previous questions and answers in the conversation.
            # it provides an easy way to add conversational QA abilities to an existing document storage/retrieval system.
            llm=llm,
            memory=memory1,
            retriever=retriever

        )

        qachat2 = ConversationalRetrievalChain.from_llm(  # ConversationalRetrievalChain is a chain in LangChain designed for conversational question answering over documents. It allows you to ask followup questions, taking into account previous questions and answers in the conversation.
            # it provides an easy way to add conversational QA abilities to an existing document storage/retrieval system.
            llm=llm,
            memory=memory2,
            retriever=retriever

        )
        while True:
            query1 = input("AI-1: ")
            response = qachat1({"question": query1})
            result = response['answer']
            print(result)

            query2 = input("AI-2: ")
            response = qachat2({"question": query2})
            result = response['answer']
            print(result)


if __name__ == "__main__":
    main()
