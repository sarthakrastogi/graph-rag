import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from knowledge_graph_rag.utils.text_preprocessing import (
    remove_stop_words_from_and_lemmatise_documents,
)


class DocumentsGraph:
    def __init__(self, documents) -> None:
        self.documents = documents
        self.preprocessed_documents = remove_stop_words_from_and_lemmatise_documents(
            documents=documents
        )
        self.G = self.create_graph_from_documents()

    def create_graph_from_documents(self):
        # Compute TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.preprocessed_documents)

        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix)

        # Create the graph
        G = nx.Graph()

        # Add nodes
        for i, doc in enumerate(self.preprocessed_documents):
            G.add_node(i, label=doc)

        # Add edges with weights (cosine similarity)
        for i in range(len(self.preprocessed_documents)):
            for j in range(i + 1, len(self.preprocessed_documents)):
                weight = cosine_sim[i, j]
                if weight > 0:  # Add edge only if there's a similarity
                    G.add_edge(i, j, weight=weight)

        return G

    def plot(self):
        # Draw the graph with labels and edge weights
        pos = nx.spring_layout(self.G)

        plt.figure(figsize=(12, 8))

        # Draw nodes with labels
        node_labels = nx.get_node_attributes(self.G, "label")
        node_labels = {
            node_number: node_label[:20] + "..."
            for node_number, node_label in node_labels.items()
        }
        nx.draw_networkx_nodes(
            self.G, pos, node_size=5000, node_color="skyblue", alpha=0.7
        )
        nx.draw_networkx_labels(
            self.G, pos, labels=node_labels, font_size=10, font_family="sans-serif"
        )

        # Draw edges with weights
        edges = self.G.edges(data=True)
        for u, v, d in edges:
            weight = d["weight"]
            nx.draw_networkx_edges(
                self.G, pos, edgelist=[(u, v)], width=weight * 10, alpha=0.5
            )
            edge_label = f"{weight:.4f}"
            mid_edge = (pos[u] + pos[v]) / 2
            plt.text(
                mid_edge[0],
                mid_edge[1],
                edge_label,
                fontsize=9,
                ha="center",
                va="center",
            )

        plt.axis("off")
        plt.show()

    def find_connected_documents(self, input_sentence, N=3):
        # Find the node corresponding to the given sentence
        input_sentence = remove_stop_words_from_and_lemmatise_documents(
            documents=[input_sentence]
        )[0]
        node_index = None
        for node, data in self.G.nodes(data=True):
            if data["label"] == input_sentence:
                node_index = node
                break

        if node_index is None:
            raise ValueError("The provided sentence is not in the graph.")

        # Get the neighbors and their edge weights
        neighbors = [
            (neighbor, self.G[node_index][neighbor]["weight"])
            for neighbor in self.G.neighbors(node_index)
        ]

        # Sort neighbors by edge weight in descending order
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # Return the top N neighbors with their full text and weights
        top_neighbors = [
            {"document": self.G.nodes[neighbor]["label"]}  # , "similarity": weight}
            for neighbor, weight in neighbors[:N]
        ]
        return top_neighbors

    def find_k_closest_sentences(self, input_sentence, N=5):
        input_sentence = remove_stop_words_from_and_lemmatise_documents(
            documents=[input_sentence]
        )[0]

        # Append the input_sentence to the list of documents
        all_docs = self.preprocessed_documents + [input_sentence]

        # Compute TF-IDF for all documents including the input sentence
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        # Compute cosine similarity between all pairs of documents
        cosine_sim = cosine_similarity(tfidf_matrix)
        similarity_scores = cosine_sim[-1, :-1]  # Exclude the similarity with itself

        # Get the indices of the top N similar documents
        closest_indices = np.argsort(similarity_scores)[-N:][::-1]

        # Return the closest N sentences and their similarity scores
        closest_sentences = [
            (self.preprocessed_documents[idx], similarity_scores[idx])
            for idx in closest_indices
            if similarity_scores[idx] > 0
        ]
        return closest_sentences

    def save(self, graph_name):
        # save graph object to file
        pickle.dump(self.G, open(f"{graph_name}.pickle", "wb"))

    def load_from_file(self, graph_name):
        # load graph object from file
        self.G = pickle.load(open(f"{graph_name}.pickle", "rb"))
