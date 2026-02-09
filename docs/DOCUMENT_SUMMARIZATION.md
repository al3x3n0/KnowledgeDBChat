Document Summarization
======================

Overview
--------

The backend can generate concise summaries for documents using the configured LLM (Ollama).
Summaries include:
- Short abstract (3–5 sentences)
- Key takeaways (5–10 bullets)
- Any dates or action items if present

Database fields
---------------

- documents.summary (text)
- documents.summary_model (string)
- documents.summary_generated_at (timestamp)

API
---

- POST `/api/v1/documents/{document_id}/summarize?force=false`
  - Triggers background summarization via Celery.
  - Returns `{ message, task_id }`.
- GET `/api/v1/documents/{document_id}`
  - Now includes `summary`, `summary_model`, `summary_generated_at` fields.

Tasks
-----

- `app.tasks.summarization_tasks.summarize_document(document_id, force=False)`

UI
--

- Documents page: “Summarize” button per document row.
- Document details modal: “Summary” section with content, timestamp, and “Regenerate”.

Notes
-----

- Input text is truncated to ~16k characters for stability.
- If full document text is missing, the first chunks are concatenated as fallback.
- Default LLM model is `settings.DEFAULT_MODEL`. Override support is wired in the service.

