from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA
import sys

PATH_DATA = "./data/diplomatic_revolution_wiki.txt"
USER_QUERY = sys.argv[1]

chunk_size = 50
chunk_overlap = 5
hf_embedding_model = "all-MiniLM-L6-v2"
retriever_search_type = "similarity"
retriever_search_kwargs = {"k":5}
ollama_model = "mistral"

loader = TextLoader(PATH_DATA)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceBgeEmbeddings(model_name=hf_embedding_model)
vectorstore = FAISS.from_documents(chunks, embedding_model)

retriever = vectorstore.as_retriever(search_type=retriever_search_type,
                                     search_kwargs=retriever_search_kwargs)

llm = ChatOllama(model=ollama_model)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

response = qa_chain.run(USER_QUERY)
print(response)