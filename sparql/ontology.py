
from config import get
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = get("sparql.endpoint")


def get_spo():
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery("""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?s ?p ?o WHERE {
            ?s ?p ?o .
            ?s a owl:Class .
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    classes = list(set(result['s']['value']
                   for result in results['results']['bindings']))
    spo = [f'{r["s"]["value"]}, {r["p"]["value"]}, {r["o"]["value"]}' for r in results['results']['bindings']]
    return classes, spo
