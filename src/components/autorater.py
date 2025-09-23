from langchain_core.documents import Document
from typing import List
from langchain.chat_models import init_chat_model

from src.utils.langsmith_loader import load_prompt

class Autorater:
    """
    determine if the reranked documents are sufficient to answer the query question
    in order to mitigate giving answer to an un-answerable question (i.e. to abstain)
    """
    def __init__(self,
                 model_autorate: str,
                 model_autorate_provider: str,
                 temperature_autorate: float,
                 top_k_autorate: int,
                 top_p_autorate: float,
                 prompt_name: str,
                 prompt_version: str):
        self._setup_prompt(prompt_name, prompt_version)
        self.autorater_llm = init_chat_model(model=model_autorate,
                                             model_provider=model_autorate_provider,
                                             temperature=temperature_autorate,
                                             top_k=top_k_autorate,
                                             top_p=top_p_autorate)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = load_prompt(prompt_name, prompt_version)
        self.autorater_prompt = prompt.format_messages()[0].content

    def autorate(self,
                 reranked_document: List[Document],
                 query: str) -> List[Document]:

        sufficient_doc = []
        for doc in reranked_document:
            try:
                response = self.autorater_llm.invoke(self.autorater_prompt.format(query, doc)).content
                score = int(response)
            except:
                score = 0
            if score == 1:
                sufficient_doc.append(doc)
        return sufficient_doc




