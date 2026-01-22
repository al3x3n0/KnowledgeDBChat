/**
 * Export all custom node components.
 */

export { default as StartNode } from './StartNode';
export { default as EndNode } from './EndNode';
export { default as ToolNode } from './ToolNode';
export { default as ConditionNode } from './ConditionNode';
export { default as ParallelNode } from './ParallelNode';
export { default as LoopNode } from './LoopNode';
export { default as WaitNode } from './WaitNode';
export { default as SwitchNode } from './SwitchNode';
export { default as SubWorkflowNode } from './SubWorkflowNode';

// Node type mapping for React Flow
export const nodeTypes = {
  startNode: require('./StartNode').default,
  endNode: require('./EndNode').default,
  toolNode: require('./ToolNode').default,
  conditionNode: require('./ConditionNode').default,
  parallelNode: require('./ParallelNode').default,
  loopNode: require('./LoopNode').default,
  waitNode: require('./WaitNode').default,
  switchNode: require('./SwitchNode').default,
  subworkflowNode: require('./SubWorkflowNode').default,
};
