"""One way to find-or-create a knowledge-graph entity."""

import pytest
from sqlalchemy import func, select

from app.models.knowledge_graph import Entity
from app.services.kg_entities import NAME_MAX, get_or_create_entity

pytestmark = pytest.mark.unit


async def _count(db, name):
    return await db.scalar(
        select(func.count()).select_from(Entity).where(Entity.canonical_name == name)
    )


async def test_the_same_name_and_type_is_one_entity(db_session):
    first, created = await get_or_create_entity(db_session, "BOLT", "tool")
    again, created_again = await get_or_create_entity(db_session, "BOLT", "tool")

    assert (created, created_again) == (True, False)
    assert first.id == again.id


async def test_a_name_longer_than_the_column_finds_its_own_row(db_session):
    # Stored truncated but looked up whole, it never matched and made a new
    # row on every call.
    long_name = "x" * (NAME_MAX + 40)

    first, _ = await get_or_create_entity(db_session, long_name, "paper")
    again, created = await get_or_create_entity(db_session, long_name, "paper")

    assert created is False
    assert again.id == first.id
    assert await _count(db_session, long_name[:NAME_MAX]) == 1


async def test_existing_duplicates_do_not_break_the_lookup(db_session):
    db_session.add_all(
        [Entity(canonical_name="dup", entity_type="tool") for _ in range(2)]
    )
    await db_session.flush()

    entity, created = await get_or_create_entity(db_session, "dup", "tool")

    assert created is False
    assert entity.canonical_name == "dup"


async def test_description_and_properties_are_stored_on_creation(db_session):
    entity, _ = await get_or_create_entity(
        db_session, "2401.00001", "paper", description="A paper", properties={"a": 1}
    )
    assert entity.description == "A paper"
    assert entity.properties == '{"a": 1}'
