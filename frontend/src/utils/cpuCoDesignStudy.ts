import type { PipelineSpec } from '../components/pipelines/PipelineGraph';
import type { ResearchCampaignDraft } from '../types';

const validateBaseline =
  'Identify the baseline and optimized INT8 attention kernel revisions, primary input shape, and replication shape. Review the expected output and check that both variants produce equivalent results. Record the input, correctness check, compiler flags, gem5 core model, and the original and enlarged load queue settings.';

const runMatrix =
  'Run the same input in gem5 with A (baseline kernel, original load queue), B (optimized kernel, original load queue), C (baseline kernel, enlarged load queue), and D (optimized kernel, enlarged load queue). Keep the core model, compiler flags, and other settings fixed. Check output equivalence and record cycles, retired instructions, load-queue-full events, configurations, and artifact references for every run. Label the simulator calls A through D.';

const analyzeAndReplicate =
  'Repeat the four configurations on the second input shape. Compare B and C with A to measure each change in isolation, then compare D with B and C for each change’s incremental effect. Report whether the ranking holds, whether the counters explain overlap, and which differences are within measurement variability. Cite evidence and identify missing or invalid measurements.';

export const CPU_CO_DESIGN_CAMPAIGN = {
  name: 'INT8 attention CPU co-design',
  goal: 'Determine whether a software optimization, a larger load queue, or both reduce simulated cycles for the INT8 attention kernel while preserving its output.',
  max_jobs: 6,
  items: [
    { title: 'Which baseline and correctness checks should the study use?', detail: validateBaseline },
    { title: 'How do the four kernel and load queue combinations compare?', detail: runMatrix },
    { title: 'Does the result hold and which change is worth pursuing?', detail: analyzeAndReplicate },
  ],
} satisfies ResearchCampaignDraft;

export const CPU_CO_DESIGN_PIPELINE: PipelineSpec = {
  name: 'cpu-co-design',
  stages: [
    {
      id: 'validate-baseline',
      goal: `${validateBaseline} Store the verified kernel and configuration references for downstream stages.`,
      contract: {
        required_finding_types: ['implementation_verified'],
        required_result_keys: ['structured_output'],
      },
    },
    {
      id: 'run-matrix',
      depends_on: ['validate-baseline'],
      goal: `Use the validated kernel and settings from the previous stage. ${runMatrix} Store the labeled A–D results in structured output.`,
      contract: {
        required_finding_type_counts: { simulated_measurement: 4 },
        required_result_keys: ['structured_output'],
      },
    },
    {
      id: 'analyze-and-replicate',
      depends_on: ['run-matrix'],
      goal: `Use the previous stage’s A–D results and settings. ${analyzeAndReplicate} Save the comparison as a formatted report.`,
      contract: {
        required_finding_type_counts: { simulated_measurement: 4 },
        required_result_keys: ['formatted_outputs'],
      },
    },
  ],
};
