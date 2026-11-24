import React from 'react';
import { Copy } from 'lucide-react';

import { apiClient } from '../../services/api';
import Button from '../common/Button';
import MCPClientSetup, { copyMCPValue } from './MCPClientSetup';

export default function MCPProfileCredentials({ profile, message, onDone }: {
  profile: { id: string; api_key: string };
  message: string;
  onDone: () => void;
}) {
  const endpoint = apiClient.getMCPEndpoint();
  return (
    <div className="space-y-3">
      <h3 className="font-medium text-gray-900" role="status">{message}</h3>
      <p className="text-sm text-gray-600">Save this API key now. It is displayed once.</p>
      {[
        { label: 'MCP endpoint', value: endpoint, action: 'Copy endpoint' },
        { label: 'API key', value: profile.api_key, action: 'Copy key' },
      ].map(({ label, value, action }) => (
        <div key={label} className="flex flex-wrap items-center gap-2">
          <label className="min-w-0 flex-1 text-sm font-medium text-gray-700">
            {label}
            <input readOnly value={value} className="mt-1 block w-full rounded-md border bg-gray-50 px-3 py-2 font-mono text-sm" />
          </label>
          <Button className="self-end" variant="secondary" onClick={() => void copyMCPValue(value, label)}>
            <Copy className="mr-2 h-4 w-4" />
            {action}
          </Button>
        </div>
      ))}
      <p className="text-sm text-gray-600">
        For manual setup, use Streamable HTTP and send the key as an Authorization Bearer token.
      </p>
      <MCPClientSetup profileId={profile.id} endpoint={endpoint} newKey={profile.api_key} />
      <Button onClick={onDone}>Done</Button>
    </div>
  );
}
