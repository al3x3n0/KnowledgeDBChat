"""Knowledge-graph tools: entities, mentions and relationships.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_entities",
        description="Search for entities (people, organizations, locations, technologies, etc.) mentioned in the knowledge base documents.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for entity names",
                },
                "entity_type": {
                    "type": "string",
                    "enum": [
                        "person",
                        "organization",
                        "location",
                        "product",
                        "technology",
                        "concept",
                        "other",
                    ],
                    "description": "Filter by entity type (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entities to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_entity_relationships",
        description="Get relationships for a specific entity, showing how it connects to other entities in the knowledge graph.",
        parameters={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of relationships to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["entity_id"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="find_documents_by_entity",
        description="Find all documents that mention a specific entity. Useful for exploring all content related to a person, organization, or concept.",
        parameters={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["entity_id"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_document_knowledge_graph",
        description="Get the knowledge graph (entities and relationships) extracted from a specific document.",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document",
                }
            },
            "required": ["document_id"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_global_knowledge_graph",
        description="Get the global knowledge graph across all documents (entities and relationships), with optional filters and limits. Useful for building an overview graph or answering questions like 'what are the key entities and how are they connected?'.",
        parameters={
            "type": "object",
            "properties": {
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity types to include (optional)",
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relationship types to include (optional)",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum relationship confidence (0.0-1.0)",
                    "default": 0.0,
                },
                "min_mentions": {
                    "type": "integer",
                    "description": "Minimum mention count for an entity to be included",
                    "default": 1,
                },
                "limit_nodes": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default: 300, max: 1000)",
                    "default": 300,
                },
                "limit_edges": {
                    "type": "integer",
                    "description": "Maximum number of edges to return (default: 1000, max: 5000)",
                    "default": 1000,
                },
                "search": {
                    "type": "string",
                    "description": "Search entity names (optional)",
                },
            },
            "required": [],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_entity_mentions",
        description="Get the document mentions for a specific entity (snippets and metadata). Useful to ground an entity in the underlying sources.",
        parameters={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of mentions to return (default: 25, max: 200)",
                    "default": 25,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)",
                    "default": 0,
                },
            },
            "required": ["entity_id"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_kg_stats",
        description="Get knowledge graph statistics: counts of entities, relationships, and mentions.",
        parameters={"type": "object", "properties": {}, "required": []},
        job_types=(),
    ),
    ToolSpec(
        name="rebuild_document_knowledge_graph",
        description="Admin-only: delete and rebuild the knowledge graph extracted from a document (re-extract entities/relationships from its chunks). Use when extraction rules/models changed or graph looks wrong.",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document",
                }
            },
            "required": ["document_id"],
        },
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="merge_entities",
        description="Admin-only: merge a source entity into a target entity (repairs duplicates). Mentions and relationships are repointed; duplicates are deduplicated.",
        parameters={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Source entity UUID to merge from",
                },
                "target_id": {
                    "type": "string",
                    "description": "Target entity UUID to merge into",
                },
            },
            "required": ["source_id", "target_id"],
        },
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="delete_entity",
        description="Admin-only: delete an entity from the knowledge graph (cascades mentions/relationships). Requires confirm_name to prevent accidental deletion.",
        parameters={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to delete"},
                "confirm_name": {
                    "type": "string",
                    "description": "Must exactly match the entity's canonical name",
                },
            },
            "required": ["entity_id", "confirm_name"],
        },
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="link_entities",
        description="Create or strengthen a relationship between two entities in the knowledge graph.",
        parameters={
            "type": "object",
            "properties": {
                "source_entity_id": {
                    "type": "string",
                    "description": "Source entity UUID",
                },
                "target_entity_id": {
                    "type": "string",
                    "description": "Target entity UUID",
                },
                "source_name": {
                    "type": "string",
                    "description": "Source entity name (alternative to ID)",
                },
                "target_name": {
                    "type": "string",
                    "description": "Target entity name (alternative to ID)",
                },
                "relationship_type": {
                    "type": "string",
                    "description": "Type of relationship (e.g., 'uses', 'extends', 'compares_to', 'improves')",
                },
                "evidence": {
                    "type": "string",
                    "description": "Evidence or explanation for this relationship",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                    "default": 0.8,
                },
            },
            "required": ["relationship_type"],
        },
        effects="write",
        job_types=("synthesis", "knowledge_expansion"),
    ),
    ToolSpec(
        name="query_kg_entities",
        description="Search for entities in the knowledge graph by name or keyword. Returns matching entities with their types and descriptions.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to match against entity names and descriptions",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Filter by entity type (e.g. person, org, location, product)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20, max 100)",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_entity_context",
        description="Get a knowledge graph entity with all its relationships and connected entities. Useful for understanding how an entity relates to others.",
        parameters={
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "UUID of the entity to get context for",
                },
            },
            "required": ["entity_id"],
        },
    ),
    ToolSpec(
        name="create_kg_entity",
        description="Create a new entity in the knowledge graph. Use this to add discovered concepts, people, organizations, or other entities during research.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical name for the entity",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Entity type: person, org, location, product, concept, technology, event, or other",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the entity",
                },
            },
            "required": ["name", "entity_type"],
        },
        effects="write",
        job_types=("research", "analysis"),
    ),
    ToolSpec(
        name="create_kg_relationship",
        description="Create a relationship between two entities in the knowledge graph. Both entities must already exist.",
        parameters={
            "type": "object",
            "properties": {
                "source_entity_id": {
                    "type": "string",
                    "description": "UUID of the source entity",
                },
                "target_entity_id": {
                    "type": "string",
                    "description": "UUID of the target entity",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Type of relationship (e.g. works_at, authored, related_to, part_of)",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0 (default 0.8)",
                },
                "evidence": {
                    "type": "string",
                    "description": "Evidence or reason for this relationship",
                },
            },
            "required": ["source_entity_id", "target_entity_id", "relation_type"],
        },
        effects="write",
        job_types=("research", "analysis"),
    ),
    ToolSpec(
        name="query_kg_graph",
        description="Query the global knowledge graph with filters. Returns nodes and edges for visualization or analysis. Useful for exploring the broader knowledge structure.",
        parameters={
            "type": "object",
            "properties": {
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Filter by entity types (e.g. ["person", "org"])',
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by relationship types",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0-1.0)",
                },
                "search": {
                    "type": "string",
                    "description": "Text search filter on entity names",
                },
                "limit_nodes": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default 50, max 200)",
                },
            },
            "required": [],
        },
    ),
)
