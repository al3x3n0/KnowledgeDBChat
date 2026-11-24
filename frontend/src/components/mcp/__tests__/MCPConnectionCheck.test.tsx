import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';

import MCPConnectionCheck from '../MCPConnectionCheck';

const endpoint = 'https://lab.example/api/v1/mcp';
const originalFetch = global.fetch;
const fetchMock = jest.fn();

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock;
});

afterEach(() => {
  global.fetch = originalFetch;
});

function rpcResponse(id: number, result: object) {
  return { ok: true, status: 200, json: async () => ({ jsonrpc: '2.0', id, result }) };
}

test('initializes the connection and discovers tools with the newly issued key', async () => {
  fetchMock.mockResolvedValueOnce(rpcResponse(1, { protocolVersion: '2025-06-18' }));
  fetchMock.mockResolvedValueOnce({ ok: true, status: 202 });
  fetchMock.mockResolvedValueOnce(rpcResponse(2, { tools: [{ name: 'search' }, { name: 'get_research_programs' }] }));
  render(<MCPConnectionCheck endpoint={endpoint} newKey="new-key" />);
  fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

  expect(await screen.findByText('Connected using MCP 2025-06-18.')).toBeInTheDocument();
  const list = screen.getByRole('list', { name: 'Discovered tools' });
  expect(within(list).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
    'search', 'get_research_programs',
  ]);
  const calls = fetchMock.mock.calls;
  expect(calls.map(([, init]) => JSON.parse(init.body).method)).toEqual([
    'initialize', 'notifications/initialized', 'tools/list',
  ]);
  for (const [url, init] of calls) {
    expect(url).toBe(endpoint);
    expect(init).toMatchObject({
      credentials: 'omit',
      redirect: 'error',
      headers: { Authorization: 'Bearer new-key', Accept: 'application/json, text/event-stream' },
    });
  }
  expect(calls[0][1].headers).not.toHaveProperty('MCP-Protocol-Version');
  expect(calls[1][1].headers['MCP-Protocol-Version']).toBe('2025-06-18');
  expect(calls[2][1].headers['MCP-Protocol-Version']).toBe('2025-06-18');
  expect(screen.queryByText('Resources')).toBeNull();
});

test('discovers research resources and prompts without retrieving their contents', async () => {
  fetchMock.mockResolvedValueOnce(rpcResponse(1, {
    protocolVersion: '2025-11-25',
    capabilities: { tools: {}, resources: {}, prompts: {} },
  }));
  fetchMock.mockResolvedValueOnce({ ok: true, status: 202 });
  fetchMock.mockResolvedValueOnce(rpcResponse(2, { tools: [{ name: 'get_research_notes' }] }));
  fetchMock.mockResolvedValueOnce(rpcResponse(3, {
    resources: [
      { name: 'Lab results', uri: 'kdbc://research-notes/123' },
      { name: 'uarch', title: 'Architecture Lab', uri: 'kdbc://research-programs/uarch.json' },
    ],
  }));
  fetchMock.mockResolvedValueOnce(rpcResponse(4, {
    prompts: [{ name: 'research_brief', title: 'Research brief' }],
  }));
  render(<MCPConnectionCheck endpoint={endpoint} newKey="new-key" />);
  fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

  const resources = await screen.findByRole('list', { name: 'Discovered resources' });
  expect(within(resources).getByText('Lab results')).toBeInTheDocument();
  expect(within(resources).getByText('Architecture Lab')).toBeInTheDocument();
  expect(within(resources).getByText('kdbc://research-programs/uarch.json')).toBeInTheDocument();
  expect(within(screen.getByRole('list', { name: 'Discovered prompts' })).getByText('Research brief')).toBeInTheDocument();
  expect(fetchMock.mock.calls.map(([, init]) => JSON.parse(init.body).method)).toEqual([
    'initialize', 'notifications/initialized', 'tools/list', 'resources/list', 'prompts/list',
  ]);
  expect(fetchMock.mock.calls[3][1].headers).toMatchObject({
    'MCP-Protocol-Version': '2025-11-25',
    Authorization: 'Bearer new-key',
  });
  expect(fetchMock.mock.calls[4][1].headers['MCP-Protocol-Version']).toBe('2025-11-25');
});

test('shows discovered tools when resource discovery fails', async () => {
  fetchMock.mockResolvedValueOnce(rpcResponse(1, {
    protocolVersion: '2025-11-25',
    capabilities: { resources: {} },
  }));
  fetchMock.mockResolvedValueOnce({ ok: true, status: 202 });
  fetchMock.mockResolvedValueOnce(rpcResponse(2, { tools: [{ name: 'search' }] }));
  fetchMock.mockResolvedValueOnce({ ok: false, status: 429 });
  render(<MCPConnectionCheck endpoint={endpoint} newKey="new-key" />);
  fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

  expect(await screen.findByRole('alert')).toHaveTextContent('Resource discovery failed.');
  expect(screen.getByRole('alert')).toHaveTextContent('The API key reached its request limit.');
  expect(screen.getByText('Connected using MCP 2025-11-25.')).toBeInTheDocument();
  expect(within(screen.getByRole('list', { name: 'Discovered tools' })).getByText('search')).toBeInTheDocument();
});

test('checks the existing profile prefix and clears the pasted key after submission', async () => {
  fetchMock.mockResolvedValueOnce({ ok: false, status: 401 });
  render(<MCPConnectionCheck endpoint={endpoint} keyPrefix="taic_profile" />);
  const input = screen.getByLabelText('API key for connection check');
  const button = screen.getByRole('button', { name: 'Test connection' });
  fireEvent.change(input, { target: { value: 'taic_other' } });
  fireEvent.click(button);
  expect(screen.getByRole('alert')).toHaveTextContent('taic_profile');
  expect(fetchMock).not.toHaveBeenCalled();

  fireEvent.change(input, { target: { value: 'taic_profile-revoked' } });
  fireEvent.click(button);
  expect(await screen.findByRole('alert')).toHaveTextContent('The API key is invalid, expired, or revoked.');
  expect(input).toHaveValue('');
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

test('reports a successful connection when discovery returns empty lists', async () => {
  fetchMock.mockResolvedValueOnce(rpcResponse(1, {
    protocolVersion: '2025-11-25',
    capabilities: { resources: {}, prompts: {} },
  }));
  fetchMock.mockResolvedValueOnce({ ok: true, status: 202 });
  fetchMock.mockResolvedValueOnce(rpcResponse(2, { tools: [] }));
  fetchMock.mockResolvedValueOnce(rpcResponse(3, { resources: [] }));
  fetchMock.mockResolvedValueOnce(rpcResponse(4, { prompts: [] }));
  render(<MCPConnectionCheck endpoint={endpoint} newKey="new-key" />);
  fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
  expect(await screen.findByText('This key currently exposes zero tools.')).toBeInTheDocument();
  expect(screen.getByText('This key currently exposes zero resources.')).toBeInTheDocument();
  expect(screen.getByText('This key currently exposes zero prompts.')).toBeInTheDocument();
});
