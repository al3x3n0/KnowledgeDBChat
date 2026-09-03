"""Folders over the document corpus.

Documents here are global — they have no `user_id` — so a folder is one user's
*view* over shared content. Most of what follows is about that: your filing
must not change what I see, and a folder you do not own must not be
distinguishable from one that does not exist.

The rest is tree arithmetic: names, depth, and the reparent that would sever a
subtree from every root.
"""

import uuid

import pytest
from sqlalchemy import select

from app.models.document import Document, DocumentSource
from app.models.document_folder import DocumentFolder
from app.models.user import User
from app.services.document_folder_service import MAX_FOLDER_DEPTH, FolderError
from app.services.document_folder_service import document_folder_service as svc

pytestmark = pytest.mark.asyncio


async def _source(db, name="src"):
    source = DocumentSource(
        id=uuid.uuid4(),
        name=f"{name}-{uuid.uuid4().hex[:6]}",
        source_type="file",
        config={},
    )
    db.add(source)
    await db.commit()
    return source


async def _document(db, source, *, title="doc", file_type="pdf", tags=None):
    doc = Document(
        id=uuid.uuid4(),
        title=title,
        content="body",
        content_hash=uuid.uuid4().hex,
        file_type=file_type,
        source_id=source.id,
        source_identifier=uuid.uuid4().hex,
        tags=tags,
    )
    db.add(doc)
    await db.commit()
    return doc


async def _other_user(db):
    user = User(
        id=uuid.uuid4(),
        username=f"other-{uuid.uuid4().hex[:6]}",
        email=f"other-{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="x",
        is_active=True,
    )
    db.add(user)
    await db.commit()
    return user


class TestAFolderIsOneUsersView:
    async def test_filing_a_document_does_not_change_another_users_view(
        self, db_session, test_user
    ):
        # The point of the join table. Documents are shared; folders are not.
        other = await _other_user(db_session)
        source = await _source(db_session)
        doc = await _document(db_session, source)

        mine = await svc.create(db_session, test_user.id, name="Mine")
        await svc.add_documents(db_session, test_user.id, mine.id, [doc.id])

        # For me it is filed; for them it is still unfiled.
        assert svc.resolve_filter("unfiled", test_user.id) is not None
        my_unfiled = await db_session.execute(
            select(Document.id).where(svc.resolve_filter("unfiled", test_user.id))
        )
        their_unfiled = await db_session.execute(
            select(Document.id).where(svc.resolve_filter("unfiled", other.id))
        )
        assert doc.id not in my_unfiled.scalars().all()
        assert doc.id in their_unfiled.scalars().all()

    async def test_another_users_folder_is_not_found_rather_than_forbidden(
        self, db_session, test_user
    ):
        other = await _other_user(db_session)
        theirs = await svc.create(db_session, other.id, name="Theirs")

        # 404, not 403: existence itself should not leak.
        with pytest.raises(FolderError) as caught:
            await svc.update(db_session, test_user.id, theirs.id, name="Renamed")
        assert caught.value.status_code == 404

    async def test_a_document_can_sit_in_several_folders(self, db_session, test_user):
        source = await _source(db_session)
        doc = await _document(db_session, source)
        a = await svc.create(db_session, test_user.id, name="A")
        b = await svc.create(db_session, test_user.id, name="B")

        await svc.add_documents(db_session, test_user.id, a.id, [doc.id])
        await svc.add_documents(db_session, test_user.id, b.id, [doc.id])

        where = await svc.folders_for_document(db_session, test_user.id, doc.id)
        assert sorted(f["name"] for f in where) == ["A", "B"]

        # Removing from one leaves the other alone.
        await svc.remove_documents(db_session, test_user.id, a.id, [doc.id])
        where = await svc.folders_for_document(db_session, test_user.id, doc.id)
        assert [f["name"] for f in where] == ["B"]


class TestFilingDocuments:
    async def test_filing_is_idempotent_and_says_what_it_found(
        self, db_session, test_user
    ):
        source = await _source(db_session)
        doc = await _document(db_session, source)
        folder = await svc.create(db_session, test_user.id, name="F")
        missing = uuid.uuid4()

        first = await svc.add_documents(db_session, test_user.id, folder.id, [doc.id])
        assert first == {"added": 1, "already_present": 0, "not_found": 0}

        again = await svc.add_documents(
            db_session, test_user.id, folder.id, [doc.id, missing]
        )
        assert again["added"] == 0
        assert again["already_present"] == 1
        assert again["not_found"] == 1

    async def test_unfiling_leaves_the_document_in_the_corpus(
        self, db_session, test_user
    ):
        source = await _source(db_session)
        doc = await _document(db_session, source)
        folder = await svc.create(db_session, test_user.id, name="F")
        await svc.add_documents(db_session, test_user.id, folder.id, [doc.id])

        assert (
            await svc.remove_documents(db_session, test_user.id, folder.id, [doc.id])
        )["removed"] == 1
        # The document itself is untouched: a folder is a view.
        assert await db_session.get(Document, doc.id) is not None

    async def test_deleting_a_folder_leaves_its_documents(self, db_session, test_user):
        source = await _source(db_session)
        doc = await _document(db_session, source)
        folder = await svc.create(db_session, test_user.id, name="F")
        await svc.add_documents(db_session, test_user.id, folder.id, [doc.id])

        await svc.delete(db_session, test_user.id, folder.id)

        assert await db_session.get(Document, doc.id) is not None
        assert await db_session.get(DocumentFolder, folder.id) is None


class TestTheShapeOfTheTree:
    async def test_names_may_repeat_across_parents_but_not_within_one(
        self, db_session, test_user
    ):
        a = await svc.create(db_session, test_user.id, name="Papers")
        b = await svc.create(db_session, test_user.id, name="Notes")

        # Same name under a different parent: fine.
        await svc.create(db_session, test_user.id, name="2026", parent_id=a.id)
        await svc.create(db_session, test_user.id, name="2026", parent_id=b.id)

        # Same name under the same parent: refused.
        with pytest.raises(FolderError) as caught:
            await svc.create(db_session, test_user.id, name="2026", parent_id=a.id)
        assert caught.value.status_code == 409

    async def test_two_root_folders_cannot_share_a_name(self, db_session, test_user):
        # The unique constraint cannot catch this: Postgres treats NULL parents
        # as distinct, so the service checks it explicitly.
        await svc.create(db_session, test_user.id, name="Root")
        with pytest.raises(FolderError) as caught:
            await svc.create(db_session, test_user.id, name="root")
        assert caught.value.status_code == 409

    async def test_a_folder_cannot_be_moved_inside_itself(self, db_session, test_user):
        parent = await svc.create(db_session, test_user.id, name="P")
        child = await svc.create(
            db_session, test_user.id, name="C", parent_id=parent.id
        )

        # Directly...
        with pytest.raises(FolderError):
            await svc.update(
                db_session, test_user.id, parent.id, parent_id=parent.id, reparent=True
            )
        # ...and through a descendant, which would sever the subtree from every
        # root while leaving valid-looking rows behind.
        with pytest.raises(FolderError):
            await svc.update(
                db_session, test_user.id, parent.id, parent_id=child.id, reparent=True
            )

    async def test_nesting_stops_at_the_limit(self, db_session, test_user):
        # MAX_FOLDER_DEPTH counts levels, and a root folder is level one, so
        # exactly that many nest successfully and the next one is refused.
        parent_id = None
        for i in range(MAX_FOLDER_DEPTH):
            folder = await svc.create(
                db_session, test_user.id, name=f"L{i}", parent_id=parent_id
            )
            parent_id = folder.id

        with pytest.raises(FolderError) as caught:
            await svc.create(
                db_session, test_user.id, name="too deep", parent_id=parent_id
            )
        assert "deep" in caught.value.detail

    async def test_a_folder_with_subfolders_needs_recursive_to_delete(
        self, db_session, test_user
    ):
        parent = await svc.create(db_session, test_user.id, name="P")
        await svc.create(db_session, test_user.id, name="C", parent_id=parent.id)

        with pytest.raises(FolderError) as caught:
            await svc.delete(db_session, test_user.id, parent.id)
        assert caught.value.status_code == 409

        result = await svc.delete(db_session, test_user.id, parent.id, recursive=True)
        assert result["subfolders_deleted"] == 1

    async def test_the_tree_nests_and_rolls_counts_up(self, db_session, test_user):
        source = await _source(db_session)
        docs = [await _document(db_session, source, title=f"d{i}") for i in range(3)]

        parent = await svc.create(db_session, test_user.id, name="Parent")
        child = await svc.create(
            db_session, test_user.id, name="Child", parent_id=parent.id
        )
        await svc.add_documents(db_session, test_user.id, parent.id, [docs[0].id])
        await svc.add_documents(
            db_session, test_user.id, child.id, [docs[1].id, docs[2].id]
        )

        tree = await svc.user_tree(db_session, test_user.id)
        assert len(tree) == 1
        root = tree[0]
        assert root["name"] == "Parent"
        assert root["document_count"] == 1
        # A closed folder's number should mean everything inside it.
        assert root["subtree_count"] == 3
        assert root["children"][0]["name"] == "Child"


class TestTheFolderKeyVocabulary:
    async def test_all_means_no_filter(self, db_session, test_user):
        assert svc.resolve_filter("all", test_user.id) is None
        assert svc.resolve_filter("", test_user.id) is None
        assert svc.resolve_filter(None, test_user.id) is None

    async def test_a_key_outside_the_vocabulary_is_refused(self, db_session, test_user):
        # A typo in a filter must not look like a successful query over
        # everything — that is how the wrong document set gets acted on.
        for bad in ("nonsense", "user:", "user:not-a-uuid", "recent:decade", "type"):
            with pytest.raises(FolderError):
                svc.resolve_filter(bad, test_user.id)

    async def test_system_folders_are_computed_from_the_documents(
        self, db_session, test_user
    ):
        source = await _source(db_session, name="GitLab")
        await _document(db_session, source, file_type="pdf")
        await _document(db_session, source, file_type="md")

        system = await svc.system_tree(db_session, test_user.id)
        keys = {node["key"] for node in system}
        assert {"all", "group:source", "group:type", "group:recent", "unfiled"} <= keys

        by_source = next(n for n in system if n["key"] == "group:source")
        assert by_source["children"][0]["key"] == f"source:{source.id}"
        assert by_source["children"][0]["document_count"] == 2

        by_type = next(n for n in system if n["key"] == "group:type")
        assert {c["name"] for c in by_type["children"]} == {"pdf", "md"}

    async def test_a_new_source_appears_without_anything_being_seeded(
        self, db_session, test_user
    ):
        # The reason system folders are computed rather than stored.
        before = await svc.system_tree(db_session, test_user.id)
        assert not any(n["key"] == "group:source" for n in before)

        source = await _source(db_session, name="ArXiv")
        await _document(db_session, source)

        after = await svc.system_tree(db_session, test_user.id)
        group = next(n for n in after if n["key"] == "group:source")
        assert group["children"][0]["document_count"] == 1

    async def test_unfiled_counts_only_what_this_user_has_not_filed(
        self, db_session, test_user
    ):
        source = await _source(db_session)
        filed = await _document(db_session, source, title="filed")
        await _document(db_session, source, title="loose")

        folder = await svc.create(db_session, test_user.id, name="F")
        await svc.add_documents(db_session, test_user.id, folder.id, [filed.id])

        system = await svc.system_tree(db_session, test_user.id)
        assert next(n for n in system if n["key"] == "unfiled")["document_count"] == 1
