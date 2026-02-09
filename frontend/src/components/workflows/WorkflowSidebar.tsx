/**
 * Workflow Sidebar Component.
 *
 * Contains:
 * - Node palette (draggable nodes)
 * - Properties panel for selected node
 */

import React, { useState } from 'react';
import {
  Play,
  Square,
  Wrench,
  GitBranch,
  GitFork,
  GitMerge,
  Repeat,
  Clock,
  ChevronDown,
  ChevronRight,
  Search,
  Download,
  Globe,
  FileText,
  Trash2,
  Upload,
  List,
  Tags,
  Bot,
  BarChart3,
  Copy,
  Network,
  Workflow,
  Plus,
  X,
} from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData, ToolParameter, SwitchCase } from './useWorkflowStore';
import InputMapper from './InputMapper';
import { ContextVariable } from './ContextAutocomplete';
import WorkflowSelector from './WorkflowSelector';

// Node type definitions for the palette
interface NodePaletteItem {
  type: string;
  label: string;
  icon: React.ReactNode;
  color: string;
  description: string;
}

const controlNodes: NodePaletteItem[] = [
  { type: 'condition', label: 'Condition', icon: <GitBranch className="w-4 h-4" />, color: 'amber', description: 'Branch based on condition' },
  { type: 'switch', label: 'Switch', icon: <GitMerge className="w-4 h-4" />, color: 'orange', description: 'Multi-way branch' },
  { type: 'parallel', label: 'Parallel', icon: <GitFork className="w-4 h-4" />, color: 'purple', description: 'Run branches in parallel' },
  { type: 'loop', label: 'Loop', icon: <Repeat className="w-4 h-4" />, color: 'cyan', description: 'Iterate over items' },
  { type: 'wait', label: 'Wait', icon: <Clock className="w-4 h-4" />, color: 'gray', description: 'Pause execution' },
  { type: 'subworkflow', label: 'Sub-Workflow', icon: <Workflow className="w-4 h-4" />, color: 'indigo', description: 'Execute another workflow' },
];

// Built-in tool nodes
const builtinTools: { name: string; label: string; icon: React.ReactNode; description: string }[] = [
  { name: 'search_documents', label: 'Search Documents', icon: <Search className="w-4 h-4" />, description: 'Search knowledge base' },
  { name: 'web_scrape', label: 'Web Scrape', icon: <Globe className="w-4 h-4" />, description: 'Scrape a wiki/portal page' },
  { name: 'ingest_url', label: 'Ingest URL', icon: <Download className="w-4 h-4" />, description: 'Scrape + save as document(s)' },
  { name: 'get_document_details', label: 'Get Document', icon: <FileText className="w-4 h-4" />, description: 'Get document details' },
  { name: 'summarize_document', label: 'Summarize', icon: <FileText className="w-4 h-4" />, description: 'Summarize a document' },
  { name: 'list_recent_documents', label: 'Recent Docs', icon: <List className="w-4 h-4" />, description: 'List recent documents' },
  { name: 'answer_question', label: 'Answer Question', icon: <Bot className="w-4 h-4" />, description: 'AI answers using docs' },
  { name: 'get_knowledge_base_stats', label: 'KB Stats', icon: <BarChart3 className="w-4 h-4" />, description: 'Get statistics' },
  { name: 'update_document_tags', label: 'Update Tags', icon: <Tags className="w-4 h-4" />, description: 'Update document tags' },
  { name: 'find_similar_documents', label: 'Find Similar', icon: <Copy className="w-4 h-4" />, description: 'Find similar documents' },
  { name: 'generate_diagram', label: 'Generate Diagram', icon: <Network className="w-4 h-4" />, description: 'Create Mermaid diagrams' },
];

// Draggable palette item
interface DraggableNodeProps {
  type: string;
  label: string;
  icon: React.ReactNode;
  builtinTool?: string;
}

const DraggableNode: React.FC<DraggableNodeProps> = ({ type, label, icon, builtinTool }) => {
  const onDragStart = (event: React.DragEvent) => {
    event.dataTransfer.setData('application/reactflow/type', type);
    event.dataTransfer.setData('application/reactflow/label', label);
    if (builtinTool) {
      event.dataTransfer.setData('application/reactflow/builtinTool', builtinTool);
    }
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      className="flex items-center space-x-2 p-2 bg-white rounded border cursor-grab hover:bg-gray-50 hover:border-gray-400 transition-colors"
      draggable
      onDragStart={onDragStart}
    >
      <div className="text-gray-600">{icon}</div>
      <span className="text-sm">{label}</span>
    </div>
  );
};

// Collapsible section
interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, children, defaultOpen = true }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b">
      <button
        className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>{title}</span>
        {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>
      {isOpen && <div className="px-3 pb-3 space-y-2">{children}</div>}
    </div>
  );
};

// Properties panel for selected node
const NodePropertiesPanel: React.FC = () => {
  const selectedNodeId = useWorkflowStore((state) => state.selectedNodeId);
  const nodes = useWorkflowStore((state) => state.nodes);
  const updateNode = useWorkflowStore((state) => state.updateNode);
  const deleteNode = useWorkflowStore((state) => state.deleteNode);
  const toolSchemas = useWorkflowStore((state) => state.toolSchemas);
  const loadToolSchemas = useWorkflowStore((state) => state.loadToolSchemas);
  const getAvailableContext = useWorkflowStore((state) => state.getAvailableContext);
  const computeContextFlow = useWorkflowStore((state) => state.computeContextFlow);
  const metadata = useWorkflowStore((state) => state.metadata);

  // Load tool schemas on mount
  React.useEffect(() => {
    loadToolSchemas();
  }, [loadToolSchemas]);

  // Recompute context flow when nodes/edges change
  React.useEffect(() => {
    computeContextFlow();
  }, [nodes, computeContextFlow]);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  if (!selectedNode) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        Select a node to view its properties
      </div>
    );
  }

  const { data } = selectedNode;
  const isProtected = selectedNode.id === 'start' || selectedNode.id === 'end';

  // Get tool schema for current node
  const toolSchema = data.builtinTool ? toolSchemas[data.builtinTool] : undefined;
  const toolParameters: ToolParameter[] = toolSchema?.parameter_list || [];

  // Get available context variables for this node
  const availableContext: ContextVariable[] = getAvailableContext(selectedNode.id);

  const handleConfigChange = (key: string, value: any) => {
    updateNode(selectedNode.id, {
      config: { ...data.config, [key]: value },
    });
  };

  return (
    <div className="p-4 space-y-4">
      {/* Node info */}
      <div>
        <label className="block text-xs font-medium text-gray-500 mb-1">Node ID</label>
        <div className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">{selectedNode.id}</div>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-500 mb-1">Type</label>
        <div className="text-sm capitalize">{data.nodeType}</div>
      </div>

      {/* Label */}
      <div>
        <label className="block text-xs font-medium text-gray-500 mb-1">Label</label>
        <input
          type="text"
          value={data.label}
          onChange={(e) => updateNode(selectedNode.id, { label: e.target.value })}
          className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
          disabled={isProtected}
        />
      </div>

      {/* Output key */}
      {data.nodeType === 'tool' && (
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Output Key</label>
          <input
            type="text"
            value={data.config.outputKey || ''}
            onChange={(e) => handleConfigChange('outputKey', e.target.value)}
            placeholder={selectedNode.id}
            className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
          />
          <p className="text-xs text-gray-400 mt-1">Variable name to store output</p>
        </div>
      )}

      {/* Condition settings */}
      {data.nodeType === 'condition' && (
        <>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Condition Type</label>
            <select
              value={data.config.condition?.type || 'truthy'}
              onChange={(e) => handleConfigChange('condition', { ...data.config.condition, type: e.target.value })}
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
            >
              <option value="truthy">Is truthy</option>
              <option value="equals">Equals</option>
              <option value="not_equals">Not equals</option>
              <option value="greater_than">Greater than</option>
              <option value="less_than">Less than</option>
              <option value="contains">Contains</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Left Value</label>
            <input
              type="text"
              value={data.config.condition?.left || ''}
              onChange={(e) => handleConfigChange('condition', { ...data.config.condition, left: e.target.value })}
              placeholder="{{context.value}}"
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 font-mono"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Right Value</label>
            <input
              type="text"
              value={data.config.condition?.right || ''}
              onChange={(e) => handleConfigChange('condition', { ...data.config.condition, right: e.target.value })}
              placeholder="Value to compare"
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
            />
          </div>
        </>
      )}

      {/* Loop settings */}
      {data.nodeType === 'loop' && (
        <>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Loop Source</label>
            <input
              type="text"
              value={data.config.loopSource || ''}
              onChange={(e) => handleConfigChange('loopSource', e.target.value)}
              placeholder="{{context.items}}"
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 font-mono"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Max Iterations</label>
            <input
              type="number"
              value={data.config.maxIterations || 100}
              onChange={(e) => handleConfigChange('maxIterations', parseInt(e.target.value))}
              min={1}
              max={1000}
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
            />
          </div>
        </>
      )}

      {/* Wait settings */}
      {data.nodeType === 'wait' && (
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Wait Duration (seconds)</label>
          <input
            type="number"
            value={data.config.waitSeconds || 0}
            onChange={(e) => handleConfigChange('waitSeconds', parseInt(e.target.value))}
            min={0}
            max={300}
            className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
          />
        </div>
      )}

      {/* Switch settings */}
      {data.nodeType === 'switch' && (
        <>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Switch Expression</label>
            <input
              type="text"
              value={data.config.switch_expression || ''}
              onChange={(e) => handleConfigChange('switch_expression', e.target.value)}
              placeholder="{{context.step1.status}}"
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 font-mono"
            />
            <p className="text-xs text-gray-400 mt-1">Value to match against cases</p>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-2">Cases</label>
            <div className="space-y-2">
              {(data.config.cases || []).map((c: { value: string; label: string }, idx: number) => (
                <div key={idx} className="flex gap-1 items-center">
                  <input
                    type="text"
                    placeholder="Value"
                    value={c.value}
                    onChange={(e) => {
                      const cases = [...(data.config.cases || [])];
                      cases[idx] = { ...cases[idx], value: e.target.value };
                      handleConfigChange('cases', cases);
                    }}
                    className="flex-1 px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 font-mono"
                  />
                  <input
                    type="text"
                    placeholder="Label"
                    value={c.label}
                    onChange={(e) => {
                      const cases = [...(data.config.cases || [])];
                      cases[idx] = { ...cases[idx], label: e.target.value };
                      handleConfigChange('cases', cases);
                    }}
                    className="flex-1 px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
                  />
                  <button
                    onClick={() => {
                      const cases = (data.config.cases || []).filter((_: any, i: number) => i !== idx);
                      handleConfigChange('cases', cases);
                    }}
                    className="p-1 text-red-500 hover:bg-red-50 rounded"
                    title="Remove case"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                onClick={() => {
                  const cases = [...(data.config.cases || []), { value: '', label: '' }];
                  handleConfigChange('cases', cases);
                }}
                className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700"
              >
                <Plus className="w-4 h-4" />
                Add Case
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-1">Non-matching values route to default</p>
          </div>
        </>
      )}

      {/* Sub-Workflow settings */}
      {data.nodeType === 'subworkflow' && (
        <>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Select Workflow</label>
            <WorkflowSelector
              value={data.config.workflow_id || ''}
              onChange={(id) => handleConfigChange('workflow_id', id)}
              excludeId={metadata.id}
            />
            <p className="text-xs text-gray-400 mt-1">Workflow to execute as sub-workflow</p>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Timeout (seconds)</label>
            <input
              type="number"
              value={data.config.timeout_seconds || 300}
              onChange={(e) => handleConfigChange('timeout_seconds', parseInt(e.target.value))}
              min={1}
              max={3600}
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">On Error</label>
            <select
              value={data.config.on_error || 'fail'}
              onChange={(e) => handleConfigChange('on_error', e.target.value)}
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
            >
              <option value="fail">Fail workflow</option>
              <option value="continue">Continue execution</option>
            </select>
          </div>
        </>
      )}

      {/* Input mapping for tool nodes */}
      {data.nodeType === 'tool' && (
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-2">Input Mapping</label>
          {toolParameters.length > 0 ? (
            <InputMapper
              parameters={toolParameters}
              inputMapping={data.config.inputMapping || {}}
              onChange={(mapping) => handleConfigChange('inputMapping', mapping)}
              availableContext={availableContext}
            />
          ) : (
            /* Fallback to JSON textarea if no schema available */
            <div>
              <p className="text-xs text-gray-400 mb-1">No schema available - use JSON format:</p>
              <textarea
                value={JSON.stringify(data.config.inputMapping || {}, null, 2)}
                onChange={(e) => {
                  try {
                    const parsed = JSON.parse(e.target.value);
                    handleConfigChange('inputMapping', parsed);
                  } catch {
                    // Invalid JSON, ignore
                  }
                }}
                placeholder='{"query": "{{context.searchTerm}}"}'
                rows={4}
                className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 font-mono"
              />
            </div>
          )}
        </div>
      )}

      {/* Delete button */}
      {!isProtected && (
        <button
          onClick={() => deleteNode(selectedNode.id)}
          className="w-full flex items-center justify-center space-x-1 px-3 py-2 bg-red-50 text-red-600 rounded border border-red-200 hover:bg-red-100"
        >
          <Trash2 className="w-4 h-4" />
          <span>Delete Node</span>
        </button>
      )}
    </div>
  );
};

// Main sidebar component
const WorkflowSidebar: React.FC = () => {
  const selectedNodeId = useWorkflowStore((state) => state.selectedNodeId);

  return (
    <div className="w-64 bg-gray-50 border-r flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b bg-white">
        <h3 className="font-medium text-sm">
          {selectedNodeId ? 'Node Properties' : 'Add Nodes'}
        </h3>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {selectedNodeId ? (
          <NodePropertiesPanel />
        ) : (
          <>
            {/* Control nodes */}
            <CollapsibleSection title="Control Flow">
              {controlNodes.map((node) => (
                <DraggableNode
                  key={node.type}
                  type={node.type}
                  label={node.label}
                  icon={node.icon}
                />
              ))}
            </CollapsibleSection>

            {/* Built-in tools */}
            <CollapsibleSection title="Document Tools">
              {builtinTools.map((tool) => (
                <DraggableNode
                  key={tool.name}
                  type="tool"
                  label={tool.label}
                  icon={tool.icon}
                  builtinTool={tool.name}
                />
              ))}
            </CollapsibleSection>

            {/* Custom tools placeholder */}
            <CollapsibleSection title="Custom Tools" defaultOpen={false}>
              <div className="text-xs text-gray-500 text-center py-2">
                Create custom tools in the Tools Manager
              </div>
            </CollapsibleSection>
          </>
        )}
      </div>

      {/* Footer hint */}
      <div className="px-3 py-2 border-t bg-white text-xs text-gray-500">
        Drag nodes to canvas • Click to edit • Delete to remove
      </div>
    </div>
  );
};

export default WorkflowSidebar;
