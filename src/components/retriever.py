from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from typing import List
from utils.reranker import Reranker
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dataclasses import dataclass

class ChunkRetriever:
    def __init__(self, retriever_search_type: str,
                 retriever_search_kwargs: dict):
        self.retriever_search_type = retriever_search_type
        self.retriever_search_kwargs = retriever_search_kwargs

    def retrieve(self, vectorstore: FAISS) -> VectorStoreRetriever:
        retriever = vectorstore.as_retriever(search_type=self.retriever_search_type,
                                             search_kwargs=self.retriever_search_kwargs)
        return retriever

class RerankRetriever(BaseRetriever):
    retriever: BaseRetriever = None
    k_rerank: int = 3
    model_rerank: str = None
    temperature_rerank: float = 0
    top_k_rerank: int = 10
    top_p_rerank: float = 0.9

    def __init__(self, retriever: BaseRetriever,
                 k_rerank: int,
                 model_rerank: str,
                 temperature_rerank: float,
                 top_k_rerank: int,
                 top_p_rerank: float):
        super().__init__()
        self.retriever = retriever
        self.k_rerank = k_rerank
        self.model_rerank = model_rerank
        self.temperature_rerank = temperature_rerank
        self.top_k_rerank = top_k_rerank
        self.top_p_rerank = top_p_rerank

    def get_relevant_documents(self, query: str) -> List[Document]:
        retrieved_docs = self.retriever.get_relevant_documents(query)
        reranked_docs = Reranker(model=self.model_rerank,
                                 temperature=self.temperature_rerank,
                                 top_k=self.top_k_rerank,
                                 top_p=self.top_p_rerank).rerank(query,
                                                                 retrieved_docs,
                                                                 top_k=self.k_rerank)
        print('# relevant doc: ',len(reranked_docs))
        return reranked_docs

