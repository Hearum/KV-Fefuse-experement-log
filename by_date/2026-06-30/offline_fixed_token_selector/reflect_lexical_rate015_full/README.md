# Reflect Offline Lexical Fixed Sets

- rate: 0.15
- tokenization: `Document: {doc}\n`, same as reflect pipeline
- no query, no draft/QK/full-attention forward; this is a pure document-intrinsic offline selector.

## Methods

- `lexical_entity_number_per_chunk`: capitalized pieces, numbers, and long alphabetic pieces.
- `lexical_idf_entity_per_chunk`: lexical entity score plus corpus token IDF.
- `lexical_idf_entity_boundary_per_chunk`: IDF/entity score plus document boundary and sentence-start prior.

| method | chunks | selected tokens |
|---|---:|---:|
| lexical_entity_number_per_chunk | 3408 | 65644 |
| lexical_idf_entity_boundary_per_chunk | 3408 | 65644 |
| lexical_idf_entity_per_chunk | 3408 | 65644 |
