from langchain_core.runnables.base import RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain.chat_models import init_chat_model

from src.components.loader import DataLoader
from src.components.chunker import TextSplitter
from src.components.embedder import DocEmbedder
from src.components.retriever import ChunkRetriever
from src.components.reranker import Reranker

class ComponentChainer:
    def __init__(self, model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float):
        self.llm = init_chat_model(model=model,
                                   model_provider=model_provider,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p)

    def chain(self, SYSTEM_PROMPT: str,
              RERANKER_PROMPT: str) -> RunnableSerializable:
        with open(SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        with open(RERANKER_PROMPT, "r", encoding="utf-8") as f:
            reranker_prompt = f.read()

        rag_prompt = ChatPromptTemplate.from_template(system_prompt)

        loader = DataLoader(metadata={"source": "battle of stalingrad"})
        splitter = TextSplitter(chunk_size=1000,
                                chunk_overlap=100
                                )
        embedder = DocEmbedder(model_name="all-MiniLM-L6-v2")
        retriever = ChunkRetriever(retriever_search_type="similarity",
                                   retriever_search_kwargs={"k":50}
                                   )
        reranker = Reranker(k_rerank=10,
                             model_rerank="gemini-2.5-flash",
                             model_rerank_provider="google_genai",
                             temperature_rerank=0,
                             top_k_rerank=5,
                             top_p_rerank=0.8,
                             reranker_prompt=reranker_prompt,
                            )

        loader_runnable = RunnableLambda(lambda x: {"question": x,
                                                    "context": loader.load_from_wikipedia_api(title="Battle of Stalingrad")})
        splitter_runnable = RunnableLambda(lambda x: {"question": x["question"],
                                                      "context": splitter.split(docs=x["context"])})
        embedder_runnable = RunnableLambda(lambda x: {"question": x["question"],
                                                      "context": embedder.embed(split_docs=x["context"],
                                                                                vs_name=None)})
        retriever_runnable = RunnableLambda(lambda x: {"question": x["question"],
                                                       "context": retriever.retrieve(vectorstore=x["context"]).invoke(x["question"])})
        reranker_runnable = RunnableLambda(lambda x: {"question": x["question"],
                                                      "context": reranker.rerank(x["context"], x["question"])})
        rag_chain = (loader_runnable.with_config(run_name='loader') |
                     splitter_runnable.with_config(run_name='splitter') |
                     embedder_runnable.with_config(run_name='embedder') |
                     retriever_runnable.with_config(run_name='retriever') |
                     reranker_runnable.with_config(run_name='reranker') |
                     rag_prompt |
                     self.llm)

        #rag_chain = (RunnableLambda(lambda x: {"question": x, "context": retriever.invoke(x)})
        #             | {
        #                 "context": RunnableLambda(lambda x: reranker.rerank(x["context"], x["question"])),
        #                 "question": RunnablePassthrough(),
        #             }
        #             | rag_prompt | self.llm
        #)
        return rag_chain

