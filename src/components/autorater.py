from langchain_core.documents import Document
from typing import List
from langchain.chat_models import init_chat_model
import asyncio

from src.utils.langsmith_loader import load_prompt
from src.utils.syncify import sync

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

    @sync
    async def autorate(self,
                 reranked_document: List[Document],
                 query: str) -> List[Document]:

        async def score_doc(query, doc):
            try:
                response = await self.autorater_llm.ainvoke(self.autorater_prompt.format(query, doc.page_content))
                score = int(response.content)
            except:
                score = 0
            return (score, doc)

        rated_tasks = [score_doc(query, doc) for doc in reranked_document]
        rated = await asyncio.gather(*rated_tasks)
        return [d[1] for d in rated if d[0] == 1]


if __name__ == "__main__":
    import os

    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    #test_doc = "model development process. Regulators should consider and amplify the importance of the following considerations and strategies when firms are establishing these foundational processes: ● Socio-technical Considerations: Managing AI risks requires a comprehensive approach that integrates both technical and social factors. NIST has identified a series of “trustworthiness characteristics” — including (1) valid and reliable, (2) safe, secure and resilient, (3) accountable and transparent, (4) explainable and interpretable, (5) privacy-enhanced, and (6) fair with harmful bias managed. According to NIST, “[c]reating trustworthy AI requires balancing each of these characteristics based on the AI system’s context of use. While all characteristics are socio-technical system attributes, accountability and transparency also relate to the processes and activities internal to an AI system and its external setting. Neglecting these characteristics can increase the probability and magnitude of",
    #test_query = "What are the six points of Socio-technical Considerations that the regulators should consider?"

    test_doc = ["the apple is in the box",
                "the banana is on the tree",
                "the apple is red"]
    test_query = "where can I find the apple?"

    model_config = {
        "model_autorate": "gemini-2.5-flash",
        "model_autorate_provider": "google_genai",
        "temperature_autorate": 0,
        "top_k_autorate": 5,
        "top_p_autorate": 0.8,
        "prompt_name": "system-autorater-prompt",
        "prompt_version": "latest"
    }

    autorater = Autorater(**model_config)
    context = autorater.autorate(reranked_document=test_doc, query=test_query)
    print(context)




