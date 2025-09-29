from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate

from src.utils.mlflow_loader import load_prompt

class LLMJudge:
    def __init__(self, model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 judge_prompt: str,
                 ):
        self._llm_judge = init_chat_model(model=model,
                                   model_provider=model_provider,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p)
        self.judge_prompt = ChatPromptTemplate.from_template(judge_prompt)

    @property
    def llm_judge(self):
        return self.judge_prompt | self._llm_judge

    @staticmethod
    def accuracy_metric(
                        outputs: dict,
                        expectations: dict,
                        llm_judge: BaseChatModel,
                        ) -> bool:
        """
        This can be replaced by mlflow.genai.scorers.Correctness
        """
        try:
            response = llm_judge.invoke({"gold_response":expectations["gold"],
                                         "llm_response":outputs["response"]}).content
            score = int(response)
        except:
            score = 0

        return bool(score)


if __name__ == "__main__":
    import os

    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

    judge_params = {
        "judge_prompt_name": "LLMJUDGE_PROMPT",
        "judge_prompt_version": "latest",
        "judge_model": "gemini-2.5-flash",
        "judge_model_provider": "google_genai",
        "judge_temperature": 0,
        "judge_top_k": 10,
        "judge_top_p": 0.9
    }

    judge_prompt = load_prompt(prompt_name=judge_params["judge_prompt_name"],
                               prompt_version=judge_params["judge_prompt_version"])
    judge_prompt_str = judge_prompt.template
    print(f"load prompt")

    dataset = load_data(name=data_name,
                        experiment_id=experiment_id,
                        tags=experiment_tags,
                        test_cases=examples)

    LLM_judge = LLMJudge(model=judge_params["judge_model"],
                         model_provider=judge_params["judge_model_provider"],
                         temperature=judge_params["judge_temperature"],
                         top_k=judge_params["judge_top_k"],
                         top_p=judge_params["judge_top_p"],
                         judge_prompt=judge_prompt_str)
    print(f"load LLM judge")

    llm_response = "The Battle of Stalingrad lasted from 17 July 1942 to 2 February 1943, which is 201 days."
    gold_response = "201 days"
    result = LLM_judge.accuracy_metric(outputs={"response":llm_response},
                                       expectations={"gold":gold_response},
                                       llm_judge=LLM_judge.llm_judge)

    print(result)
