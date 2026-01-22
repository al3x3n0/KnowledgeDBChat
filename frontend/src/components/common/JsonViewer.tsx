import React from 'react';

type NodeProps = {
  data: any;
  label?: string;
  depth?: number;
};

const Toggle: React.FC<{ open: boolean; onClick: () => void }> = ({ open, onClick }) => (
  <button className="inline-flex items-center text-xs text-gray-600 hover:text-gray-900 mr-1" onClick={onClick}>
    {open ? '−' : '+'}
  </button>
);

const isObject = (v: any) => v && typeof v === 'object' && !Array.isArray(v);

const Node: React.FC<NodeProps> = ({ data, label, depth = 0 }) => {
  const [open, setOpen] = React.useState(depth < 1);

  if (data === null || typeof data !== 'object') {
    return (
      <div className="whitespace-pre-wrap break-words">
        {label ? <span className="text-gray-700 mr-1">{label}:</span> : null}
        <span className="text-gray-900">{String(data)}</span>
      </div>
    );
  }

  const entries = Array.isArray(data) ? data.map((v, i) => [String(i), v] as const) : Object.entries(data);

  return (
    <div>
      <div className="flex items-center">
        <Toggle open={open} onClick={() => setOpen(o => !o)} />
        {label && <span className="text-gray-700 mr-1">{label}:</span>}
        <span className="text-gray-500">{Array.isArray(data) ? `[${entries.length}]` : `{${entries.length}}`}</span>
      </div>
      {open && (
        <div className="ml-4 border-l pl-2 space-y-1">
          {entries.map(([k, v]) => (
            <Node key={k} data={v} label={k} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
};

const JsonViewer: React.FC<{ json: any }> = ({ json }) => {
  return (
    <div className="text-xs">
      <Node data={json} depth={0} />
    </div>
  );
};

export default JsonViewer;

