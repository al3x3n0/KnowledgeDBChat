#!/usr/bin/env python3
"""
Script to ingest OptimizeIR seed documents into the knowledge base.

Usage:
    python seed_data/ingest_seed_data.py

This script will:
1. Create a document source for OptimizeIR documentation
2. Load all markdown documents from seed_data/documents/
3. Process them into chunks and add to the vector store
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from uuid import uuid4

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set environment to use existing .env
os.chdir(backend_dir)


async def main():
    # Import after path setup
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.database import AsyncSessionLocal
    from app.models.document import Document, DocumentChunk, DocumentSource
    from app.services.vector_store import vector_store_service
    from app.services.text_processor import TextProcessor
    text_processor = TextProcessor()

    DOCUMENTS_DIR = Path(__file__).parent / "documents"
    SOURCE_NAME = "OptimizeIR Documentation"
    SOURCE_TYPE = "seed_data"

    print("=" * 60)
    print("OptimizeIR Seed Data Ingestion")
    print("=" * 60)

    async with AsyncSessionLocal() as db:
        # Check if source already exists
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.name == SOURCE_NAME)
        )
        source = result.scalar_one_or_none()

        if source:
            print(f"\nSource '{SOURCE_NAME}' already exists (ID: {source.id})")
            print("Updating existing documents...")
        else:
            # Create document source
            source = DocumentSource(
                id=uuid4(),
                name=SOURCE_NAME,
                source_type=SOURCE_TYPE,
                config={
                    "description": "Seed data for OptimizeIR LLVM-based optimization toolkit",
                    "auto_generated": True,
                },
                is_active=True,
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)
            print(f"\nCreated source: {SOURCE_NAME} (ID: {source.id})")

        # Initialize vector store
        print("\nInitializing vector store...")
        await vector_store_service.initialize()

        # Process each markdown file
        md_files = sorted(DOCUMENTS_DIR.glob("*.md"))
        print(f"\nFound {len(md_files)} documents to process")

        for md_file in md_files:
            print(f"\n{'─' * 50}")
            print(f"Processing: {md_file.name}")

            # Read content
            content = md_file.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Extract title from first heading
            title = md_file.stem.replace("_", " ").title()
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Check if document already exists
            result = await db.execute(
                select(Document).where(
                    Document.source_id == source.id,
                    Document.source_identifier == md_file.name,
                )
            )
            existing_doc = result.scalar_one_or_none()

            if existing_doc:
                if existing_doc.content_hash == content_hash:
                    print(f"  → Unchanged, skipping")
                    continue
                else:
                    # Update existing document
                    print(f"  → Updating existing document")
                    existing_doc.title = title
                    existing_doc.content = content
                    existing_doc.content_hash = content_hash
                    document = existing_doc

                    # Delete old chunks
                    for chunk in document.chunks:
                        await db.delete(chunk)
                    await db.commit()

                    # Delete from vector store
                    await vector_store_service.delete_document_chunks(document.id)
            else:
                # Create new document
                document = Document(
                    id=uuid4(),
                    source_id=source.id,
                    source_identifier=md_file.name,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    file_type="text/markdown",
                    is_processed=False,
                )
                db.add(document)
                await db.commit()
                await db.refresh(document)
                print(f"  → Created document: {document.id}")

            # Split into chunks
            chunks_data = await text_processor.split_text(
                content,
                chunk_size=1000,
                chunk_overlap=200,
            )
            print(f"  → Split into {len(chunks_data)} chunks")

            # Create chunk records
            chunks = []
            for idx, chunk_text in enumerate(chunks_data):
                chunk = DocumentChunk(
                    id=uuid4(),
                    document_id=document.id,
                    content=chunk_text,
                    content_hash=hashlib.sha256(chunk_text.encode()).hexdigest(),
                    chunk_index=idx,
                )
                chunks.append(chunk)
                db.add(chunk)

            await db.commit()

            # Add to vector store
            await vector_store_service.add_document_chunks(document, chunks)
            print(f"  → Added to vector store")

            # Mark as processed
            document.is_processed = True
            await db.commit()

        print(f"\n{'=' * 60}")
        print("Ingestion complete!")
        print(f"{'=' * 60}")

        # Print summary
        from sqlalchemy import func
        doc_count_result = await db.execute(
            select(func.count()).select_from(Document).where(Document.source_id == source.id)
        )
        doc_count = doc_count_result.scalar()

        chunk_count_result = await db.execute(
            select(func.count()).select_from(DocumentChunk)
            .join(Document)
            .where(Document.source_id == source.id)
        )
        total_chunks = chunk_count_result.scalar()

        print(f"\nSource: {SOURCE_NAME}")
        print(f"Documents: {doc_count}")
        print(f"Total chunks: {total_chunks}")
        print(f"\nTemplates available in: seed_data/templates/")
        print("  - project_report_template.docx")
        print("  - technical_specification_template.docx")
        print("  - executive_summary_template.docx")
        print("  - release_notes_template.docx")


if __name__ == "__main__":
    asyncio.run(main())
