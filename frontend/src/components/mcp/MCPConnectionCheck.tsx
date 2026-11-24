import React, { useState } from 'react';

import { checkMCPConnection } from '../../services/mcpConnection';
import type { MCPConnectionResult } from '../../services/mcpConnection';
import Button from '../common/Button';

type CheckState =
  | { status: 'idle' | 'checking' }
  | { status: 'error'; message: string }
  | ({ status: 'connected' } & MCPConnectionResult);

export default function MCPConnectionCheck({ endpoint, newKey, keyPrefix }: {
  endpoint: string;
  newKey?: string;
  keyPrefix?: string;
}) {
  const [enteredKey, setEnteredKey] = useState('');
  const [check, setCheck] = useState<CheckState>({ status: 'idle' });

  async function test(event: React.FormEvent) {
    event.preventDefault();
    const credential = newKey ?? enteredKey.trim();
    if (!credential || check.status === 'checking') return;
    if (keyPrefix && !credential.startsWith(keyPrefix)) {
      setCheck({ status: 'error', message: `This profile’s API key starts with ${keyPrefix}.` });
      return;
    }
    setEnteredKey('');
    setCheck({ status: 'checking' });
    try {
      setCheck({ status: 'connected', ...await checkMCPConnection(endpoint, credential) });
    } catch (error) {
      setCheck({ status: 'error', message: (error as Error).message });
    }
  }

  return (
    <form className="space-y-3 border-t pt-4" onSubmit={test}>
      <p className="text-sm text-gray-700">Check whether this browser can connect and discover the profile’s tools, resources, and prompts.</p>
      {!newKey && (
        <label className="block text-sm font-medium text-gray-700">
          API key for connection check
          <input
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={enteredKey}
            onChange={(event) => {
              setEnteredKey(event.target.value);
              setCheck({ status: 'idle' });
            }}
            disabled={check.status === 'checking'}
            className="mt-1 block w-full max-w-lg rounded-md border bg-white px-3 py-2"
          />
        </label>
      )}
      <Button variant="secondary" type="submit" loading={check.status === 'checking'} disabled={!newKey && !enteredKey.trim()}>
        Test connection
      </Button>
      {check.status === 'error' && <p role="alert" className="text-sm text-red-700">{check.message}</p>}
      {check.status === 'connected' && (
        <div role="status" className="space-y-2 rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-900">
          <p className="font-medium">Connected using MCP {check.protocolVersion}.</p>
          <div>
            <p className="font-medium">Tools</p>
            {check.tools.length ? (
              <ul aria-label="Discovered tools" className="list-inside list-disc">
                {check.tools.map((tool) => <li key={tool}><code>{tool}</code></li>)}
              </ul>
            ) : <p>This key currently exposes zero tools.</p>}
          </div>
          {check.resources.status === 'listed' && (
            <div className="border-t border-green-200 pt-2">
              <p className="font-medium">Resources</p>
              {check.resources.items.length ? (
                <ul aria-label="Discovered resources" className="max-h-48 space-y-1 overflow-y-auto">
                  {check.resources.items.map(({ name, uri }) => (
                    <li key={uri}>
                      <span className="block">{name}</span>
                      <code className="break-all text-xs">{uri}</code>
                    </li>
                  ))}
                </ul>
              ) : <p>This key currently exposes zero resources.</p>}
            </div>
          )}
          {check.resources.status === 'failed' && (
            <p role="alert" className="text-amber-900">Resource discovery failed. {check.resources.message}</p>
          )}
          {check.prompts.status === 'listed' && (
            <div className="border-t border-green-200 pt-2">
              <p className="font-medium">Prompts</p>
              {check.prompts.items.length ? (
                <ul aria-label="Discovered prompts" className="list-inside list-disc">
                  {check.prompts.items.map(({ id, name }) => <li key={id}>{name}</li>)}
                </ul>
              ) : <p>This key currently exposes zero prompts.</p>}
            </div>
          )}
          {check.prompts.status === 'failed' && (
            <p role="alert" className="text-amber-900">Prompt discovery failed. {check.prompts.message}</p>
          )}
        </div>
      )}
    </form>
  );
}
