from src.helper import load_pdf, textt_split, download_hugging_face_embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_community.llms import CTransformers
from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# print(PINECONE_API_KEY)


extracted_data = load_pdf('data')   
text_chunks = textt_split(extracted_data)
embedding = download_hugging_face_embedding()

print('Downloaded embedding model')

# print(text_chunks)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="aimed"
pc.list_indexes()
docsearc=PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embedding,index_name=index_name)




