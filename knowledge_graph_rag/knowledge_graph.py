from knowledge_graph_rag.utils.llm import llm_call
import json
import re
from tqdm.notebook import tqdm
import networkx as nx
from collections import defaultdict, deque
import matplotlib.pyplot as plt

from .utils.prompts import knowledge_graph_creation_system_prompt


class KnowledgeGraph:
    def __init__(self, documents) -> None:
        self.documents = documents

    def remove_trailing_commas(self, json_string):
        # Remove trailing commas from JSON arrays and objects
        json_string = re.sub(r",\s*([\]}])", r"\1", json_string)
        return json_string

    def create_knowledge_representations(self, documents):
        knowledge_representations_of_individual_documents = []
        for document in tqdm(documents):
            messages = [
                {"role": "system", "content": knowledge_graph_creation_system_prompt},
                {"role": "user", "content": document},
            ]

            response = llm_call(messages=messages)
            response = response.lower()
            response = self.remove_trailing_commas(response)
            knowledge_representations_of_individual_documents.append(
                json.loads(response)
            )

        return knowledge_representations_of_individual_documents

    def create_knowledge_graph_from_representations(self, representations):
        G = nx.DiGraph()

        def add_edge(source, target, relationship):
            if G.has_edge(source, target):
                G[source][target]["relationship"] += f", {relationship}"
                G[source][target]["weight"] = G[source][target].get("weight", 1) + 1
            else:
                G.add_edge(source, target, relationship=relationship, weight=1)

        for rep in representations:
            for item in rep:
                source = item["entity"]
                if "connections" in item:
                    for conn in item["connections"]:
                        target = conn["entity"]
                        relationship = conn["relationship"]
                        add_edge(source, target, relationship)

        return G

    def create(self):
        self.knowledge_representations = self.create_knowledge_representations(
            self.documents
        )
        self.G = self.create_knowledge_graph_from_representations(
            self.knowledge_representations
        )

    def plot(self):
        pos = nx.spring_layout(self.G)
        plt.figure(figsize=(12, 8))

        # Draw nodes with labels
        nx.draw_networkx_nodes(
            self.G, pos, node_size=5000, node_color="skyblue", alpha=0.7
        )
        node_labels = {
            node: node[:20] + "..." if len(node) > 20 else node
            for node in self.G.nodes()
        }
        nx.draw_networkx_labels(
            self.G, pos, labels=node_labels, font_size=10, font_family="sans-serif"
        )

        # Draw edges with weights
        edges = self.G.edges(data=True)
        for u, v, d in edges:
            weight = d.get("weight", 1)  # Default to 1 if weight is not present
            nx.draw_networkx_edges(
                self.G, pos, edgelist=[(u, v)], width=weight, alpha=0.5
            )

            # Add edge labels (relationship and weight)
            edge_label = (
                f"{d['relationship'][:20]}...\n(w:{weight:.2f})"
                if len(d["relationship"]) > 20
                else f"{d['relationship']}\n(w:{weight:.2f})"
            )
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            plt.text(
                x,
                y,
                edge_label,
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def search_document(self, input_document, max_depth=3):
        knowledge_representations_of_input_document = (
            self.create_knowledge_representations(documents=[input_document])
        )
        result = []
        for rep in knowledge_representations_of_input_document:
            for item in rep:
                source_entity = item["entity"]
                if source_entity in self.G:
                    result.append(f"\nEntity: {source_entity}")
                    result.extend(self.bfs_traversal(source_entity, max_depth))
        return "\n".join(result)

    def bfs_traversal(self, start_node, max_depth):
        visited = set()
        queue = deque([(start_node, 0)])
        result = []
        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                break
            if node not in visited:
                visited.add(node)
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        relationship = self.G[node][neighbor]["relationship"]
                        result.append(f"  -> {neighbor} (Relationship: {relationship})")
                        queue.append((neighbor, depth + 1))
        return result
