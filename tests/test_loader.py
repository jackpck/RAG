import pytest

from src.components.loader import DataLoader
import data.test.test_page_content_examples as test_examples

pdf_path = "data/pdfs/wp_generative_ai_risk_management_in_fs.pdf"

@pytest.fixture
def my_pdf():
    loader = DataLoader(metadata={})
    doc = loader.load_from_pdf(pdf_path=pdf_path,
                               split_from_mid=True)
    return doc

def test_page_content(my_pdf):
    """
    spot check content on specific pages
    """
    assert test_examples.page_content_7 in my_pdf[6].page_content
    assert test_examples.page_content_11 in my_pdf[10].page_content
    assert test_examples.page_content_12 in my_pdf[11].page_content


