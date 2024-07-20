from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embedding
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_community.llms import CTransformers
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from src.prompts import *
from dotenv import load_dotenv

app=Flask(__name__)
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

embedding=download_hugging_face_embedding()
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="aimed"


docsearch=PineconeVectorStore.from_existing_index(index_name,embedding)
PROMPT=PromptTemplate(template=prompt_template,input_vaiables=['context','question'])
chain_type_kwargs={'prompt':PROMPT}

config = {'max_new_tokens': 256, 'repetition_penalty': 0.8}
llm = CTransformers(model='model//llama-2-7b-chat.ggmlv3.q4_0.bin',model_type='llama',config=config)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    chain_type_kwargs=chain_type_kwargs
)


@app.route('/')

def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080 ,debug=True)





