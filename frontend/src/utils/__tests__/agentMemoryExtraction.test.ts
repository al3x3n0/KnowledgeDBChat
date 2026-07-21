import {
  normalizeJobMemoryExtractionSummary,
  normalizeManualExtractionResult,
  normalizeJobMemoryPersistenceSummary,
} from '../agentMemoryExtraction';

describe('agentMemoryExtraction utils', () => {
  test('normalizes extraction summary from runtime payload', () => {
    const summary = normalizeJobMemoryExtractionSummary({
      status: 'completed',
      created_count: '3',
      parsed_count: 5,
      skipped_duplicates: '2',
      is_relaunch_chain: 1,
      relaunch_root_job_id: '  abc  ',
    });

    expect(summary).not.toBeNull();
    expect(summary?.status).toBe('completed');
    expect(summary?.created_count).toBe(3);
    expect(summary?.parsed_count).toBe(5);
    expect(summary?.skipped_duplicates).toBe(2);
    expect(summary?.is_relaunch_chain).toBe(true);
    expect(summary?.relaunch_root_job_id).toBe('abc');
  });

  test('accepts api extraction shape with memories_created', () => {
    const summary = normalizeJobMemoryExtractionSummary({
      memories_created: 4,
      candidate_count: '6',
    });

    expect(summary).not.toBeNull();
    expect(summary?.created_count).toBe(4);
    expect(summary?.candidate_count).toBe(6);
  });

  test('returns null for empty or invalid payloads', () => {
    expect(normalizeJobMemoryExtractionSummary(null)).toBeNull();
    expect(normalizeJobMemoryExtractionSummary(undefined)).toBeNull();
    expect(normalizeJobMemoryExtractionSummary({})).toBeNull();
  });

  test('normalizes manual extraction result with completed status', () => {
    const summary = normalizeManualExtractionResult({
      job_id: 'job-1',
      memories_created: 2,
      parsed_count: 3,
      candidate_count: 3,
      skipped_duplicates: 1,
      is_relaunch_chain: false,
      relaunch_root_job_id: null,
      memories: [],
    });

    expect(summary.status).toBe('completed');
    expect(summary.created_count).toBe(2);
    expect(summary.skipped_duplicates).toBe(1);
  });

  test('normalizes memory persistence summary with runtime and extraction', () => {
    const summary = normalizeJobMemoryPersistenceSummary({
      enabled: 1,
      injected_count: '4',
      runtime: {
        profile: ' balanced ',
        role: 'researcher',
        limit: '12',
      },
      extraction: {
        status: 'completed',
        memories_created: 3,
        skipped_duplicates: '1',
      },
    });

    expect(summary).not.toBeNull();
    expect(summary?.enabled).toBe(true);
    expect(summary?.injected_count).toBe(4);
    expect(summary?.runtime?.profile).toBe('balanced');
    expect(summary?.runtime?.limit).toBe(12);
    expect(summary?.extraction?.created_count).toBe(3);
    expect(summary?.extraction?.skipped_duplicates).toBe(1);
  });

  test('returns null for empty memory persistence summary', () => {
    expect(normalizeJobMemoryPersistenceSummary(null)).toBeNull();
    expect(normalizeJobMemoryPersistenceSummary({})).toBeNull();
    expect(normalizeJobMemoryPersistenceSummary({ runtime: {} })).not.toBeNull();
  });
});
