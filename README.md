This one is a QnA Chatbot created using Langchain and streamlit for web interface.
I have used PyPDF2 to extract text from pdf that are uploaded, LangChain's RecursiveCharacterTextSplitter to split them into chunks.
I have created embeddings by using GoogleGenerativeAIEmbeddings.
I have used FAISS (Facebook AI Similarity Search) to  create a vector store for similarity search.
Then, I have used ChatGoogleGenerativeAI as a conversational model .
load_qa_chain method is used to load the chain.
The API key used here is Google API Key ,
