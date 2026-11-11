import { removePipelineStages, removeStageToolConfiguration, replacePipelineStage, stageBindingSourceIssues } from '../pipelineEditing';

it('renames graph and binding references atomically without changing result paths or goals', () => {
  const source = { name: 'p', stages: [
    { id: 'produce', goal: 'produce' },
    { id: 'consume', depends_on: ['produce'], may_revisit: ['produce'], tool_bindings: { check: { code: { stage: 'produce', path: '/produce/code' } } } },
  ] };
  const next = replacePipelineStage(source, 'produce', { ...source.stages[0], id: 'renamed' });
  expect(next.stages?.[1].depends_on).toEqual(['renamed']);
  expect(next.stages?.[1].may_revisit).toEqual(['renamed']);
  expect(next.stages?.[1].tool_bindings?.check.code).toEqual({ stage: 'renamed', path: '/produce/code' });
  expect(next.stages?.[0].goal).toBe('produce');
  expect(source.stages[1].depends_on).toEqual(['produce']);
});

it('refuses empty ids and collisions rather than merging stages', () => {
  const source = { stages: [{ id: 'one' }, { id: 'two' }] };
  for (const id of ['', 'two']) {
    expect(() => replacePipelineStage(source, 'one', { id })).toThrow('nonempty and unique');
  }
  expect(source.stages.map((stage) => stage.id)).toEqual(['one', 'two']);
});

it('removes deleted graph and revisit references while preserving explicit bindings for repair', () => {
  const bound = { read: { query: { stage: 'produce', path: '/query' } } };
  const source = { name: 'p', stages: [
    { id: 'produce' }, { id: 'other' },
    { id: 'consume', depends_on: ['produce', 'other'], may_revisit: ['produce', 'other'],
      tool_bindings: bound, tool_defaults: { read: { limit: 3 } } },
  ] };
  const next = removePipelineStages(source, ['produce']);
  expect(next.stages?.map((stage) => stage.id)).toEqual(['other', 'consume']);
  expect(next.stages?.[1]).toMatchObject({
    depends_on: ['other'], may_revisit: ['other'], tool_bindings: bound, tool_defaults: { read: { limit: 3 } },
  });
  expect(stageBindingSourceIssues(next.stages![1], ['other', 'consume'])).toEqual([
    "read.query: source stage 'produce' no longer exists. Choose another dependency or explicitly remove this binding.",
  ]);
  expect(source.stages[2].depends_on).toEqual(['produce', 'other']);
  expect(source.stages[2].may_revisit).toEqual(['produce', 'other']);
});

it('distinguishes a removed dependency from a deleted binding source', () => {
  const stage = { id: 'consume', depends_on: [], tool_bindings: { read: { query: { stage: 'produce', path: '' } } } };
  expect(stageBindingSourceIssues(stage, ['produce', 'consume'])).toEqual([
    "read.query: source stage 'produce' is not a dependency. Restore the dependency or change this binding.",
  ]);
  expect(stageBindingSourceIssues({ ...stage, depends_on: ['produce'] }, ['produce', 'consume'])).toEqual([]);
  expect(removePipelineStages({ stages: [{ id: 'a' }, { id: 'b' }] }, ['a', 'b']).stages).toEqual([]);
});

it.each([undefined, null, [], ['removed', 'keep']])('removes one tool configuration while preserving access mode %j', (tools) => {
  const reference = { stage: 'produce', path: '/value' };
  const source = { id: 'consume', goal: 'Consume', tools,
    tool_defaults: { removed: { value: false }, keep: { value: 0 } },
    tool_bindings: { removed: { value: reference }, keep: { value: reference } },
  };
  const next = removeStageToolConfiguration(source, 'removed');
  expect(next.tools).toEqual(Array.isArray(tools) ? tools.filter((name) => name !== 'removed') : tools);
  expect(next.tool_defaults).toEqual({ keep: { value: 0 } });
  expect(next.tool_bindings).toEqual({ keep: { value: reference } });
  expect(next.goal).toBe('Consume');
  expect(source.tool_defaults.removed).toEqual({ value: false });
  expect(source.tool_bindings.removed).toEqual({ value: reference });
});

it('removes empty argument maps without changing evidence-only access to all tools', () => {
  const stage = { id: 'consume', tools: ['removed'], tool_defaults: { removed: {} }, tool_bindings: { removed: {} } };
  const next = removeStageToolConfiguration(stage, 'removed');
  expect(next.tools).toEqual([]);
  expect(next.tool_defaults).toBeUndefined();
  expect(next.tool_bindings).toBeUndefined();
});
