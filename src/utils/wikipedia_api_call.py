import wikipedia

class WikipediaContent:
    def __init__(self, title: str):
        self.title = title

    def get_content(self) -> str:
        content = wikipedia.WikipediaPage(self.title).content
        return content