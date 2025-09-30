from langchain_core.documents import Document
from typing import List
from langchain.chat_models import init_chat_model
import asyncio

from src.utils.langsmith_loader import load_prompt
from src.utils.syncify import syncify

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

    @syncify
    async def autorate(self,
                 reranked_document: List[str],
                 query: str) -> List[str]:

        async def score_doc(doc):
            try:
                response = await self.autorater_llm.ainvoke(self.autorater_prompt.format(query, doc)).content
                score = int(response)
            except:
                score = 0
            return (score, doc)

        rated_tasks = [score_doc(doc) for doc in reranked_document]
        rated = await asyncio.gather(*rated_tasks)
        return [d[1] for d in rated if d[0] == 1]




