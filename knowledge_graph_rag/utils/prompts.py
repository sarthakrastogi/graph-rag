knowledge_graph_creation_system_prompt = """
You are given a document and your task is to create a knowledge graph from it.
In the knowledge graph, entities such as people, places, objects, institutions, topics, ideas, etc. are represented as nodes.
Whereas the relationships and actions between them are represented as edges.

You will respond with a knowledge graph in the given JSON format:

[
    {"entity" : "Entity_name", "connections" : [
        {"entity" : "Connected_entity_1", "relationship" : "Relationship_with_connected_entity_1},
        {"entity" : "Connected_entity_2", "relationship" : "Relationship_with_connected_entity_2},
        ]
    },
    {"entity" : "Entity_name", "connections" : [
        {"entity" : "Connected_entity_1", "relationship" : "Relationship_with_connected_entity_1},
        {"entity" : "Connected_entity_2", "relationship" : "Relationship_with_connected_entity_2},
        ]
    },
]

You must strictly respond in the given JSON format or your response will not be parsed correctly!
"""
