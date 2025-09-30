from typing import List
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.vectorstores.base import VectorStoreRetriever
import time

from src.utils.langsmith_loader import load_prompt

class Reranker:
    k_rerank: int = 3
    model_rerank: str = None
    model_rerank_provider: str = None
    temperature_rerank: float = 0
    top_k_rerank: int = 1
    top_p_rerank: float = 0.9
    prompt_name: str
    prompt_version: str

    def __init__(self,
                 k_rerank: int,
                 model_rerank: str,
                 model_rerank_provider: str,
                 temperature_rerank: float,
                 top_k_rerank: int,
                 top_p_rerank: float,
                 prompt_name: str,
                 prompt_version: str):
        self.k_rerank = k_rerank
        self.rerank_llm = init_chat_model(model=model_rerank,
                                          model_provider=model_rerank_provider,
                                          temperature=temperature_rerank,
                                          top_k=top_k_rerank,
                                          top_p=top_p_rerank)
        self._setup_prompt(prompt_name, prompt_version)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = load_prompt(prompt_name, prompt_version)
        self.reranker_prompt = prompt.format_messages()[0].content

    def rerank(self, retriever: VectorStoreRetriever,
               query: str) -> List[Document]:
        retrieved_docs = retriever.invoke(query)

        ranked = []
        for doc in retrieved_docs:
            try:
                response = self.rerank_llm.invoke(self.reranker_prompt.format(query,
                                                                    doc.page_content)).content
                score = int(response)
            except:
                score = 0
            ranked.append((score, doc))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.k_rerank]]


