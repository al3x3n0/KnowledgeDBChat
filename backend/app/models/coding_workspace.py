"""A coding workspace that outlives the job that made it.

Until now a workspace was a `tempfile.mkdtemp()` directory recorded in a
process-local dict, created inside the celery worker that ran the job. Three
things followed from that, and all three block any view onto the work:

* **The API could not see it.** `manager.get(workspace_id)` from a request
  handler returns None -- always, not intermittently -- because the dict
  belongs to a different process in a different container.
* **It did not survive the worker.** A restart, a reload, a crash, and the
  directory is gone with everything in it.
* **It was cleaned up on completion**, with only changed files pushed to
  object storage. So a benchmark's workspace no longer exists by the time
  anyone asks how the number was produced.

That last one is the reason this row exists at all. For a research pipeline
the environment is part of the evidence: a measurement whose workspace has
been deleted is a number nobody can re-derive, and re-deriving it is the whole
point of recording it.

What is stored here is the *identity and shape* of a workspace, not its
contents -- the files live on a volume both the API and the workers mount.
`state` carries the parts of `CodingWorkspace` that cannot be recovered by
looking at the directory: which files it started with (so a diff is possible)
and what checkpoints were taken.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class CodingWorkspaceRecord(Base):
    """One workspace, addressable from any process."""

    __tablename__ = "coding_workspaces"

    #: The workspace id the agent's tools already use. Not a new identifier:
    #: `state["coding_workspace_id"]`, the tool parameters and the job results
    #: all name this, and inventing a second one would leave every existing
    #: reference pointing at nothing.
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    #: The job that created it. Kept as a plain column rather than a foreign
    #: key with a cascade: when the job is deleted the workspace becomes
    #: orphaned, not invalid, and losing the record of how a measurement was
    #: produced is the wrong repair for deleting a run.
    owner_job_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    #: Where the files are, on the shared volume. Absolute, because the two
    #: processes that read it mount the volume at the same path.
    base_path = Column(Text, nullable=False)

    #: Where it came from, for a reader who did not launch it.
    source_id = Column(String(200), nullable=True)
    repo_url = Column(Text, nullable=True)
    branch = Column(String(200), nullable=True)

    #: The parts of the in-memory workspace that reading the directory cannot
    #: recover: the hashes of the files it started with, and the checkpoints
    #: taken. Without the first there is no way to say what the run CHANGED,
    #: which is the question a reviewer actually asks.
    state = Column(JSON, nullable=True)

    #: `active` while a job holds it, `retained` once the job is done and the
    #: files are being kept for inspection, `discarded` when the directory has
    #: been removed. The row outlives the directory deliberately -- knowing a
    #: workspace existed and is gone is different from never having heard of
    #: it.
    status = Column(String(24), nullable=False, default="active", index=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_used_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (Index("ix_coding_workspaces_user_status", "user_id", "status"),)

    def __repr__(self):
        return f"<CodingWorkspaceRecord(id={self.id}, status={self.status})>"
