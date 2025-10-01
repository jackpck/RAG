import os
import pytest

from src.components.autorater import Autorater

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()


@pytest.fixture()
def my_doc():
    test_doc = ["the apple is in the box",
                "the banana is on the tree",
                "the apple is red"]
    return test_doc

@pytest.fixture()
def my_query():
    test_query = "where can I find the apple?"
    return test_query


def test_autorater(my_doc, my_query):
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
    context = autorater.autorate(reranked_document=my_doc,
                                 query=my_query)
    assert context == ['the apple is in the box']




