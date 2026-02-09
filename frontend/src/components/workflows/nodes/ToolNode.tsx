/**
 * Tool node - executes a custom or built-in tool.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Wrench, Search, Globe, Download, FileText, Trash2, Upload, List, Tags, Bot } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

// Icon mapping for built-in tools
const toolIcons: Record<string, React.ReactNode> = {
  search_documents: <Search className="w-5 h-5" />,
  web_scrape: <Globe className="w-5 h-5" />,
  ingest_url: <Download className="w-5 h-5" />,
  get_document_details: <FileText className="w-5 h-5" />,
  summarize_document: <FileText className="w-5 h-5" />,
  delete_document: <Trash2 className="w-5 h-5" />,
  list_recent_documents: <List className="w-5 h-5" />,
  request_file_upload: <Upload className="w-5 h-5" />,
  update_document_tags: <Tags className="w-5 h-5" />,
  answer_question: <Bot className="w-5 h-5" />,
};

const ToolNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  const icon = data.builtinTool
    ? toolIcons[data.builtinTool] || <Wrench className="w-5 h-5" />
    : <Wrench className="w-5 h-5" />;

  const toolName = data.builtinTool
    ? data.builtinTool.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
    : data.label;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[160px] cursor-pointer transition-all ${
        selected
          ? 'border-blue-500 ring-2 ring-blue-300 bg-blue-50'
          : 'border-blue-300 hover:border-blue-400 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-blue-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <div className="text-blue-600">{icon}</div>
        <div>
          <div className="font-medium text-sm text-gray-800">{toolName}</div>
          {data.config.outputKey && (
            <div className="text-xs text-gray-500">
              Output: {data.config.outputKey}
            </div>
          )}
        </div>
      </div>

      {/* Source handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-blue-500 !border-2 !border-white"
      />
    </div>
  );
};

export default ToolNode;
