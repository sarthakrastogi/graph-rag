class Document:
    def __init__(self, content, embedding=[], title="", source="") -> None:
        self.content = content
        self.embedding = embedding
        self.title = title
        self.source = source
