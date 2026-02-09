# Research Lab Pilot Runbook (AI Hub + Research Hub)

Goal: demonstrate a complete “happy path” loop for three research-native workflows:

- Datasets → Train → Model Registry (adapter appears)
- Real-time training progress in UI
- Eval templates (plugin-gated) + “Use in Chat” to apply the trained adapter

## 0) Enable the pilot plugin bundle

In the Admin UI:

- Go to `Admin` → `AI Hub`
- Click **Research Lab (All Workflows)**

This enables:

- Dataset presets: `perf_regression_triage_v1`, `repro_checklist_v1`, `gap_analysis_hypotheses_v1`
- Eval templates: `perf_regression_triage_v1`, `extraction_quality_v1`, `literature_triage_v1`

If you prefer env config (no Admin changes), use:

- `AI_HUB_EVAL_ENABLED_TEMPLATE_IDS=perf_regression_triage_v1,extraction_quality_v1,literature_triage_v1`
- `AI_HUB_DATASET_ENABLED_PRESET_IDS=perf_regression_triage_v1,repro_checklist_v1,gap_analysis_hypotheses_v1`

## 1) Workflow A — Perf/Investigation Triage

1. Create a Reading List with relevant docs (prior triage notes, benchmark writeups, postmortems).
2. AI Hub → `Datasets` → **Generate (Presets)**:
   - Preset: `Perf Regression Triage (v1)`
   - Enable **Auto-validate**
   - Enable **Continue to Training**
3. AI Hub → `Training`:
   - Confirm the Create Job wizard is prefilled for the generated dataset.
   - Create the job and watch real-time progress in the training list.
4. AI Hub → `Models`:
   - Confirm the new adapter appears.
   - Deploy if needed.
   - Run Eval:
     - Template: `Perf Regression Triage (v1)`
   - Click **Use in Chat** to start a session pinned to the deployed adapter.

## 2) Workflow B — Evidence / Experiment Extraction

1. Make a Reading List with experimental reports (papers, internal notes, log summaries).
2. AI Hub → `Datasets` → **Generate (Presets)**:
   - Preset: `Repro Checklist (v1)` (good default for turning messy notes into repeatable structure)
3. Train → Model appears in registry.
4. AI Hub → `Models`:
   - Deploy adapter.
   - Run Eval:
     - Template: `Evidence Extraction Quality (v1)`
   - Use in Chat:
     - Paste an excerpt from a report and ask for structured extraction + evidence quotes.

## 3) Workflow C — Literature Triage + Gap Analysis & Hypotheses

1. Create a Reading List of new papers (or internal summaries).
2. Research Hub:
   - Run synthesis type `Gap Analysis & Hypotheses`
   - Save as a Research Note
3. AI Hub → `Datasets` → **Generate (Presets)**:
   - Preset: `Gap Analysis & Hypotheses (v1)`
4. Train → Model appears in registry.
5. AI Hub → `Models`:
   - Deploy adapter.
   - Run Eval:
     - Template: `Literature Triage (v1)`
   - Use in Chat:
     - Ask for “read/skim/skip + unknowns + follow-ups” against a new abstract.

## What to collect during the pilot (lightweight)

For each workflow, record:

- 3 “good” examples and 3 “bad” examples (what went wrong)
- 1 metric baseline and after (time-to-triage, extraction correctness, triage precision@k)
- A short list of rubric changes needed (these become new plugin templates)

