from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from typing import List
from utils.reranker import Reranker
from langchain_core.documents import Document

class ChunkRetriever:
    def __init__(self, retriever_search_type: str,
                 retriever_search_kwargs: dict):
        self.retriever_search_type = retriever_search_type
        self.retriever_search_kwargs = retriever_search_kwargs

    def retrieve(self, vectorstore: FAISS) -> VectorStoreRetriever:
        retriever = vectorstore.as_retriever(search_type=self.retriever_search_type,
                                             search_kwargs=self.retriever_search_kwargs)
        return retriever

class RerankRetriever:
    def __init__(self, base_retriever: VectorStoreRetriever,
                 top_k: int):
        self.base_retriever = base_retriever
        self.top_k = top_k
        self.reranker = Reranker

    def get_relevant_documents(self, query: str) -> List[Document]:
        retrieved_docs = self.base_retriever.get_relevant_documents(query)
        reranked_docs = self.reranker.rerank(query,
                                         retrieved_docs,
                                         top_k=self.top_k)
        return reranked_docs

