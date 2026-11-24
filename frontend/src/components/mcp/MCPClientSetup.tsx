import React, { useState } from 'react';
import { Copy } from 'lucide-react';
import toast from 'react-hot-toast';

import Button from '../common/Button';
import MCPConnectionCheck from './MCPConnectionCheck';

const clients = {
  copilot: {
    name: 'GitHub Copilot',
    destination: '~/.copilot/mcp-config.json',
    documentation: 'https://code.visualstudio.com/docs/agents/reference/mcp-configuration',
  },
  cursor: {
    name: 'Cursor',
    destination: '.cursor/mcp.json',
    documentation: 'https://cursor.com/docs/mcp',
  },
};

type Client = keyof typeof clients;

export async function copyMCPValue(value: string, label: string) {
  try {
    await navigator.clipboard.writeText(value);
    toast.success(`${label} copied`);
  } catch {
    toast.error('Clipboard unavailable. Select and copy the value manually.');
  }
}

export default function MCPClientSetup({ profileId, endpoint, newKey, keyPrefix }: {
  profileId: string;
  endpoint: string;
  newKey?: string;
  keyPrefix?: string;
}) {
  const [client, setClient] = useState<Client>('copilot');
  const profileSuffix = profileId.replace(/[^a-z0-9]/gi, '').slice(0, 12);
  const serverName = `knowledgeDb${profileSuffix}`;
  const keyVariable = `KDBC_MCP_KEY_${profileSuffix.toUpperCase()}`;
  const reference = client === 'copilot' ? keyVariable : `env:${keyVariable}`;
  const connection = {
    url: endpoint,
    headers: { Authorization: 'Bearer ${' + reference + '}' },
  };
  const server = client === 'copilot'
    ? { type: 'http', ...connection, tools: ['*'] }
    : connection;
  const config = JSON.stringify({ mcpServers: { [serverName]: server } }, null, 2);
  const selected = clients[client];

  return (
    <details className="rounded-md border bg-gray-50 p-4">
      <summary className="cursor-pointer font-medium text-gray-900">Client setup</summary>
      <div className="mt-3 space-y-3">
        <label className="flex flex-wrap items-center gap-2 text-sm font-medium text-gray-700">
          MCP client
          <select
            className="rounded-md border bg-white px-3 py-2 text-sm"
            value={client}
            onChange={(event) => setClient(event.target.value as Client)}
          >
            {Object.entries(clients).map(([id, { name }]) => <option key={id} value={id}>{name}</option>)}
          </select>
        </label>
        <p className="text-sm text-gray-700">
          Add this configuration to <code>{selected.destination}</code>. If the file already
          exists, merge the server entry into its <code>mcpServers</code> object.
        </p>
        <p className="text-sm text-gray-700">
          Set <code className="break-all">{keyVariable}</code> to your profile’s API key
          in the environment where {selected.name} runs, then restart the client.
          {' '}<a href={selected.documentation} target="_blank" rel="noreferrer" className="text-primary-700 underline">Client documentation</a>
        </p>
        <textarea
          aria-label={`${selected.name} MCP configuration`}
          className="block w-full rounded-md border bg-white p-3 font-mono text-xs text-gray-800"
          readOnly
          rows={config.split('\n').length}
          spellCheck={false}
          value={config}
        />
        <Button variant="secondary" onClick={() => void copyMCPValue(config, 'MCP configuration')}>
          <Copy className="mr-2 h-4 w-4" />
          Copy configuration
        </Button>
        <MCPConnectionCheck endpoint={endpoint} newKey={newKey} keyPrefix={keyPrefix} />
      </div>
    </details>
  );
}
