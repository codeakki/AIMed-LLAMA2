from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#load files from directory
def load_pdf(data):
   loader= DirectoryLoader(data,glob='*.pdf',loader_cls=PyMuPDFLoader)

   documents=loader.load()  
   return documents



#Covert Data into chunks | Create text chunks

def textt_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



#download embedding model 
def download_hugging_face_embedding():
    embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding




