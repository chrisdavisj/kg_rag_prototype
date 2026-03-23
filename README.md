# KG RAG Prototype

`kg_rag_prototype` is a conceptual proof of concept for experimenting with retrieval-augmented generation over a knowledge graph exposed through SPARQL.

The project explores a simple but useful research question:

Can we turn a natural-language query into a compact, semantically relevant graph context by combining:
- ontology-aware entity selection,
- embedding-based similarity,
- resource matching against a live knowledge graph,
- bounded graph expansion, and
- lightweight context pruning?

This repository is not intended to be a production app. It is a research-oriented prototype that makes the pipeline easy to inspect, modify, and compare across retrieval strategies.

## Why This Exists

Traditional RAG pipelines usually retrieve chunks of text from vector stores or search indexes. This project instead treats a knowledge graph as the retrieval substrate.

The central idea is:

1. identify ontology classes or graph concepts related to a user query,
2. use those classes to find relevant graph resources,
3. expand the local neighborhood around those resources,
4. compress that neighborhood into a context payload that could be passed to an LLM.

That makes this repository useful for experiments around:
- graph-native retrieval,
- ontology-guided grounding,
- semantic filtering of RDF resources,
- explainable context construction,
- comparing entity extraction strategies before downstream generation.

## Current Status

This codebase should be understood as an experiment scaffold.

What it already provides:
- a readable end-to-end retrieval pipeline,
- interchangeable similarity and NER-based matching strategies,
- SPARQL-driven graph exploration,
- a minimal configuration layer,
- clear module boundaries for further iteration.

What it is not trying to be:
- a hosted application,
- a polished SDK,
- a benchmark suite,
- a production-ready KG RAG system.

## Pipeline Overview

The main orchestration function lives in [main.py](/Users/chrisdavisj/kg_rag_prototype/main.py).

Given a prompt, the intended workflow is:

1. Load ontology classes from the configured SPARQL endpoint.
2. Extract the graph concepts most semantically related to the prompt.
3. Estimate how many graph hops are worth exploring for the selected concepts.
4. Retrieve candidate resources for those concepts from the graph.
5. Filter the matched resources based on semantic relevance.
6. Expand graph neighborhoods around the filtered resources.
7. Prune the resulting triples into a compact context string for downstream LLM use.

In short, the prototype turns:

`user question -> candidate graph concepts -> relevant resources -> local graph neighborhood -> LLM-ready context`

## Repository Structure

```text
kg_rag_prototype/
├── main.py
├── config.py
├── config.yaml
├── embeddings/
│   └── embedder.py
├── filters/
│   ├── similarity.py
│   ├── resource_filter.py
│   ├── threshold.py
│   ├── spacy_ner_similarity_func.py
│   ├── flair_ner_similarity_func.py
│   ├── stanza_ner_similarity_func.py
│   └── huggingface_models_ner_similarity_func.py
├── sparql/
│   ├── ontology.py
│   ├── hops.py
│   ├── matcher.py
│   └── expander.py
└── utils/
    ├── pruning.py
    └── url_replacer.py
```

## Component Guide

### `main.py`

Coordinates the full KG RAG pipeline.

### `config.py` and `config.yaml`

Provide static configuration plus runtime overrides for experimental tuning.

### `embeddings/embedder.py`

Initializes the sentence-transformer model used across semantic matching stages.

### `filters/`

Contains the semantic selection layer:
- `similarity.py` implements the default prompt-to-class similarity workflow.
- `resource_filter.py` narrows candidate resources after matching.
- `threshold.py` provides a dynamic threshold heuristic.
- the NER-based modules provide alternative concept extraction strategies using SpaCy, Flair, Stanza, or Hugging Face.

### `sparql/`

Contains graph-facing logic:
- `ontology.py` retrieves ontology classes and triples,
- `hops.py` estimates contextual graph depth,
- `matcher.py` finds resources associated with matched classes,
- `expander.py` constructs local RDF subgraphs around matched resources.

### `utils/`

Contains post-processing helpers:
- `pruning.py` is the place for graph-to-context compression,
- `url_replacer.py` is an optional hook for replacing linked URLs with fetched content.

## Intended Experimental Workflow

This prototype is best used as a lab bench for trying retrieval ideas rather than as a finished interface.

A typical experiment cycle looks like:

1. point the system at a SPARQL-accessible knowledge graph,
2. choose or swap a similarity strategy,
3. run a prompt through the pipeline,
4. inspect which ontology classes and resources were selected,
5. examine the expanded triples,
6. refine thresholds, hop settings, or filtering logic,
7. compare the resulting context quality for downstream LLM tasks.

## Configuration

The default configuration lives in [config.yaml](/Users/chrisdavisj/kg_rag_prototype/config.yaml).

Example fields:

```yaml
sparql:
  endpoint: "https://your-endpoint.example/sparql"

thresholds:
  preferred_context_hops: 3
  min_hops_to_be_explored: 3
  max_hops_threshold: 5
  preferred_confidence: 0.23

model:
  name: "all-MiniLM-L6-v2"
  use_cuda_if_available: true

multi_threading:
  num_workers_to_be_used: 10
```

Important notes:
- `sparql.endpoint` must point to a live SPARQL endpoint containing ontology classes and instance data.
- `preferred_confidence` controls how aggressively semantic matches are accepted.
- hop-related settings influence the size and breadth of retrieved context.
- the embedding model can be swapped to test retrieval quality and performance tradeoffs.

## Setup

This repository does not yet ship with a pinned dependency file, but the intended environment is standard Python 3 with the following libraries:

```text
sentence-transformers
torch
SPARQLWrapper
rdflib
pyyaml
requests
spacy
flair
stanza
transformers
```

Depending on which similarity strategy you use, not all optional NLP libraries are required at once.

Suggested setup flow:

1. Create a Python 3 virtual environment.
2. Install the required dependencies for your chosen experiment path.
3. Update `config.yaml` with a reachable SPARQL endpoint.
4. Ensure any optional model assets required by SpaCy, Flair, Stanza, or Transformers are available.

## Running the Prototype

The simplest entry point is:

```bash
python3 main.py
```

You will be prompted for a natural-language query, and the current pipeline will return a textual representation of the pruned graph context.

Example experiment prompts:
- `What research organizations work on renewable energy storage?`
- `Which people are connected to the Apollo program?`
- `Find context related to machine learning conferences in Europe.`

## Swapping Class Selectors

The default pipeline uses the class-selection function in [filters/similarity.py](/Users/chrisdavisj/kg_rag_prototype/filters/similarity.py), but you can pass an alternative selector into `kg_rag_agent()` when running programmatically.

Example:

```python
from main import kg_rag_agent
from filters.spacy_ner_similarity_func import spacy_similarity_func

result = kg_rag_agent(
    "Which organizations are connected to renewable energy storage?",
    class_selector=spacy_similarity_func,
)

print(result)
```

Available selector modules include:
- [filters/similarity.py](/Users/chrisdavisj/kg_rag_prototype/filters/similarity.py) for the default embedding-based class selector
- [filters/spacy_ner_similarity_func.py](/Users/chrisdavisj/kg_rag_prototype/filters/spacy_ner_similarity_func.py)
- [filters/flair_ner_similarity_func.py](/Users/chrisdavisj/kg_rag_prototype/filters/flair_ner_similarity_func.py)
- [filters/stanza_ner_similarity_func.py](/Users/chrisdavisj/kg_rag_prototype/filters/stanza_ner_similarity_func.py)
- [filters/huggingface_models_ner_similarity_func.py](/Users/chrisdavisj/kg_rag_prototype/filters/huggingface_models_ner_similarity_func.py)

Each selector is expected to follow this shape:

```python
def some_selector(prompt: str, ontology_classes: list[str]) -> list[str]:
    ...
```

Notes:
- the default selector is the lightest option for experimentation
- SpaCy, Flair, Stanza, and Hugging Face variants may load larger NLP models at import time
- Stanza may require downloading language resources before first use
- different selectors may use different selection policies such as thresholding or `top_k`

## What Success Looks Like

In the context of this project, success is not measured by polished UX. It is measured by whether the pipeline can produce graph-grounded context that is:
- semantically relevant to the query,
- compact enough for downstream prompting,
- structurally richer than plain document retrieval,
- understandable and debuggable by a researcher.

## Known Gaps

As a proof of concept, several parts are intentionally lightweight or still evolving:
- the pruning stage is currently a placeholder for more intelligent context compression,
- dependency installation is not yet standardized,
- evaluation and benchmarking are not yet built in,
- error handling and logging are minimal,
- some retrieval heuristics are best treated as hypotheses rather than final design choices.

These gaps are acceptable for the project’s current goal: validating the retrieval concept and learning where graph-based RAG adds value.

## Research Directions

Natural next experiments for this repository include:
- comparing ontology-class retrieval against document-only RAG baselines,
- swapping embedding models and threshold strategies,
- ranking triples by prompt relevance before context assembly,
- adding label-aware matching rather than URI-only similarity,
- generating structured prompt packs instead of raw triple text,
- evaluating answer quality with and without graph expansion,
- integrating graph paths as explicit reasoning traces for an LLM.

## Design Principles

This prototype is organized around a few simple principles:
- keep the pipeline transparent,
- prefer modular experiments over framework complexity,
- make retrieval stages individually swappable,
- treat the knowledge graph as a first-class retrieval source,
- optimize for learning, not product completeness.

## Contributing

Contributions are most helpful when they improve one of these areas:
- retrieval quality,
- pruning and compression,
- SPARQL query robustness,
- reproducibility,
- experimental evaluation,
- documentation and usage clarity.

If you extend the project, prefer small, inspectable changes that preserve the visibility of each stage in the pipeline.

## Summary

`kg_rag_prototype` is an experimental KG-RAG scaffold for studying how semantic search, ontology structure, and SPARQL-driven graph expansion can work together to build LLM context from a knowledge graph.

It is best read as a promising research prototype: ambitious in concept, intentionally modular in implementation, and designed to support iteration as the experiment matures.

# License
**Code** in this repository (including scripts and implementations) is licensed under the [Apache License 2.0](./LICENSE).

By contributing, you agree to license your contributions under these same terms.
