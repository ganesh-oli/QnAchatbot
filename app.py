import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()                                         #loads the environment variable here
os.getenv("GOOGLE_API_KEY")                           #fetches the value of Api key in this environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  #For a model "genai", the function "configure", configures 
                                                      #the model with that api key

def get_pdf_text(pdf_docs):
    text=""                              #suruma empty let gareko, pachi extract gareko text tya bascha.
    for pdf in pdf_docs:                 #loops in pdf_docs
        pdf_reader= PdfReader(pdf)       #EXTRACTS TEXT from PdfReader() which is a part of Library " PyPDF2"
        for page in pdf_reader.pages:    #extract gareko pdf ko each page ma loop 
            text+= page.extract_text()   #text vanne variable ma each page bata extract vayera store hudai garcha.
    return text                          #append vayeko text return huncha .


#dividing those texts into chunks of texts
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    #RecursiveCharacterTextSplitter method from langchain.text_splitter . Splitting into chunk size of 10000 
    #and overlap of 1000 characters
    chunks = text_splitter.split_text(text) #split_text method called to split the above extracted texts
    return chunks                           #returns the splitted chunks



#creating a vectore stores to represent Q n A as vectors , and embeddings too
#which helps for similarity search to find the most relevant answer for the query
def get_vector_store(text_chunks): #text_chunks contains list of chunks of texts
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") #create embedding through GoogleGenaAIEmbeddings, Model specifies path of embedding model trained earlier.
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) 
    #Creating a vectore Store using FAISS ( Facebook AI SImilarity Search )
    #Each text chunk is now converted to vector representation through the above embedding model.
    vector_store.save_local("faiss_index")
    #A vector store with name "faiss_index" is locally created and save on the computer.It remains there until deleted.


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])





def main():
    st.set_page_config("Chat with perisha PDF")
    st.header("OLI's chat")

    user_question = st.text_input("Ask your questions")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()