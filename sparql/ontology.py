from config import get_required
from errors import SPARQLQueryError
from SPARQLWrapper import SPARQLWrapper, JSON


def get_spo():
    sparql = SPARQLWrapper(get_required("sparql.endpoint"))
    sparql.setQuery("""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?s ?p ?o WHERE {
            ?s ?p ?o .
            ?s a owl:Class .
        }
    """)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as exc:
        raise SPARQLQueryError("Failed to load ontology classes from SPARQL endpoint") from exc

    bindings = results.get("results", {}).get("bindings")
    if not isinstance(bindings, list):
        raise SPARQLQueryError("Ontology query returned an invalid SPARQL payload")

    classes = list({result["s"]["value"] for result in bindings if "s" in result})
    spo = [
        f'{r["s"]["value"]}, {r["p"]["value"]}, {r["o"]["value"]}'
        for r in bindings
        if all(key in r for key in ("s", "p", "o"))
    ]
    return classes, spo
