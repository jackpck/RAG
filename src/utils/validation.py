from langchain.chat_models import init_chat_model

from src.utils.langsmith_loader import load_prompt

class LLMJudge:
    def __init__(self, model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 prompt_name: str,
                 prompt_version: str):
        self.llm_judge = init_chat_model(model=model,
                                   model_provider=model_provider,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p)
        self._setup_prompt(prompt_name, prompt_version)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = load_prompt(prompt_name, prompt_version)
        self.judge_prompt = prompt.format_messages()[0].content

    def accuracy_metric(self, inputs: dict,
                        outputs: dict,
                        reference_outputs: dict,
                        ) -> bool:
        try:
            response = self.llm_judge.invoke(self.judge_prompt.format(reference_outputs["gold"],
                                                                outputs["response"])).content
            score = int(response)
        except:
            score = 0

        return bool(score)


if __name__ == "__main__":
    import os

    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
    os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

    model = "gemini-2.5-flash"
    model_provider = "google_genai"
    temperature = 0
    top_k = 10
    top_p = 0.9
    judge_prompt_name = "system-llmjudge-prompt"
    judge_prompt_version = "latest"

    LLM_judge = LLMJudge(model=model,
                         model_provider=model_provider,
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p,
                         prompt_name=judge_prompt_name,
                         prompt_version=judge_prompt_version)

    llm_response = "The Battle of Stalingrad lasted from 17 July 1942 to 2 February 1943, which is 201 days."
    gold_response = "201 days"
    result = LLM_judge.accuracy_metric(inputs={},
                                       outputs={"response":llm_response},
                                       reference_outputs={"gold":gold_response})

    print(result)
