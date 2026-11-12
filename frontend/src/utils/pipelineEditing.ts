import type { PipelineSpec, PipelineStageSpec } from '../components/pipelines/PipelineGraph';

/** Explicitly remove one tool's extra selection and arguments in one write. */
export function removeStageToolConfiguration(stage: PipelineStageSpec, name: string): PipelineStageSpec {
  const defaults = { ...(stage.tool_defaults || {}) };
  const bindings = { ...(stage.tool_bindings || {}) };
  delete defaults[name];
  delete bindings[name];
  return {
    ...stage,
    ...(Array.isArray(stage.tools) && { tools: stage.tools.filter((tool) => tool !== name) }),
    tool_defaults: Object.keys(defaults).length ? defaults : undefined,
    tool_bindings: Object.keys(bindings).length ? bindings : undefined,
  };
}

/** Remove graph references, retaining bound inputs for explicit repair. */
export function removePipelineStages(spec: PipelineSpec, removed: readonly string[]): PipelineSpec {
  const gone = new Set(removed);
  return {
    ...spec,
    stages: (spec.stages || []).filter((stage) => !gone.has(stage.id)).map((stage) => ({
      ...stage,
      ...(Array.isArray(stage.depends_on) && { depends_on: stage.depends_on.filter((id) => !gone.has(id)) }),
      ...(Array.isArray(stage.may_revisit) && { may_revisit: stage.may_revisit.filter((id: string) => !gone.has(id)) }),
    })),
  };
}

/** Explain missing binding sources even when tool inspection is collapsed. */
export function stageBindingSourceIssues(stage: PipelineStageSpec, stageIds: readonly string[]): string[] {
  const bindings = stage.tool_bindings;
  if (!bindings || typeof bindings !== 'object' || Array.isArray(bindings)) return [];
  return Object.entries(bindings).flatMap(([tool, references]) => {
    if (!references || typeof references !== 'object' || Array.isArray(references)) return [];
    return Object.entries(references).flatMap(([argument, reference]) => {
      const source = reference?.stage;
      const name = `${tool}.${argument}`;
      if (!source || typeof source !== 'string') return [`${name}: choose a dependency stage for this binding.`];
      if (!stageIds.includes(source)) return [`${name}: source stage '${source}' no longer exists. Choose another dependency or explicitly remove this binding.`];
      if (!(stage.depends_on || []).includes(source)) return [`${name}: source stage '${source}' is not a dependency. Restore the dependency or change this binding.`];
      return [];
    });
  });
}

/** Replace a stage and rename its structural references in the same spec write. */
export function replacePipelineStage(spec: PipelineSpec, previousId: string, next: PipelineStageSpec): PipelineSpec {
  const stages = spec.stages || [];
  if (next.id !== previousId && (!next.id.trim() || stages.some((stage) => stage.id === next.id))) {
    throw new Error('Stage id must be nonempty and unique.');
  }
  const rename = (id: string) => id === previousId ? next.id : id;
  return {
    ...spec,
    stages: stages.map((original) => {
      const stage = original.id === previousId ? next : original;
      if (next.id === previousId) return stage;
      const bindings = stage.tool_bindings;
      return {
        ...stage,
        ...(stage.depends_on && { depends_on: stage.depends_on.map(rename) }),
        ...(Array.isArray(stage.may_revisit) && { may_revisit: stage.may_revisit.map(rename) }),
        ...(bindings && typeof bindings === 'object' && !Array.isArray(bindings) && {
          tool_bindings: Object.fromEntries(Object.entries(bindings).map(([tool, refs]) => [tool,
            refs && typeof refs === 'object' && !Array.isArray(refs)
              ? Object.fromEntries(Object.entries(refs).map(([argument, ref]) => [argument,
                ref && typeof ref === 'object' ? { ...ref, stage: rename(ref.stage) } : ref,
              ])) : refs,
          ])),
        }),
      };
    }),
  };
}
