from setuptools import setup, find_packages

with open("README.MD", "r") as f:
    readme_content = f.read()

setup(
    name="knowledge_graph_rag",
    version="0.1.0",
    packages=find_packages(),
    long_description=readme_content,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.24.0",
        "networkx==3.2.1",
        "nltk==3.8.1",
        "litellm==1.34.0",
    ],
)
