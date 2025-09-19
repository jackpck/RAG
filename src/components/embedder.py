from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List
from langchain_core.documents import Document
from langsmith import traceable

class DocEmbedder:
    def __init__(self, model_name: str):
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name)

    @traceable
    def embed(self, split_docs: List[Document]) -> FAISS:
        vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        return vectorstore
