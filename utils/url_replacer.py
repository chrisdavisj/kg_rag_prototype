from rdflib import Graph
import requests
import re


# Used to replace the real content in place, in case of it being linked to a source rather than the source being in the KG
def replace_urls_with_content(graph: Graph):
    for s, p, o in graph:
        if isinstance(o, str) and re.match(r'^https?://', o):
            try:
                response = requests.get(o, timeout=5)
                if response.status_code == 200:
                    graph.remove((s, p, o))
                    graph.add((s, p, response.text))
            except Exception:
                continue
    return graph
