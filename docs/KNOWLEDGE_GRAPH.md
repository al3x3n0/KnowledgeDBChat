Knowledge Graph Overview
========================

This backend adds a lightweight knowledge graph built from uploaded documents.

- Entities: canonical nodes (person, org, location, product, email, url, other)
- Mentions: occurrences of entities in specific documents/chunks with offsets
- Relationships: directed edges between entities with provenance and evidence

Extraction
----------

During document processing, after chunking and vector indexing, the service
extracts entities and simple relations from each chunk using regex-based
heuristics and stores them in PostgreSQL tables.

API Endpoints
-------------

- `GET /kg/stats` — counts of entities, relationships, mentions
- `GET /kg/document/{document_id}/graph` — nodes/edges for a single document
- `GET /kg/entities?q=&limit=&offset=` — list/search entities
- `GET /kg/entity/{entity_id}` — get entity details
- `PATCH /kg/entity/{entity_id}` — update entity (admin)
- `GET /kg/entity/{entity_id}/relationships` — relationships touching the entity
- `POST /kg/document/{document_id}/rebuild` — delete and re-extract KG for a document
- `GET /kg/entity/{entity_id}/mentions?limit=&offset=` — paginated mentions with totals
- `DELETE /kg/entity/{entity_id}?confirm_name=...` — delete entity (admin)
- `POST /kg/entities/merge` — merge entities (admin)
- `GET /kg/chunk/{chunk_id}?evidence=...` — fetch chunk with evidence offsets
- `GET /kg/audit?action=&user_id=&date_from=&date_to=&limit=&offset=` — KG audit logs (admin)

Notes
-----

- The extractor is intentionally lightweight; you can extend
  `backend/app/services/knowledge_extraction.py` to use your preferred NER/RE models.
- Schema and services are designed to be compatible with async SQLAlchemy.
