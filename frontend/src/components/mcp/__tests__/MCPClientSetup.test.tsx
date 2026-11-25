import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import MCPClientSetup from '../MCPClientSetup';

test('shows the native Copilot and Cursor formats and copies the selected configuration', async () => {
  const writeText = jest.fn().mockResolvedValue(undefined);
  const clipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
  try {
    render(
      <MCPClientSetup
        profileId="abcdef12-3456-7890-abcd-ef1234567890"
        endpoint="https://lab.example/api/v1/mcp"
      />
    );
    fireEvent.click(screen.getByText('Client setup'));
    const copilot = screen.getByRole('textbox', { name: 'GitHub Copilot MCP configuration' }) as HTMLTextAreaElement;
    expect(JSON.parse(copilot.value)).toEqual({
      mcpServers: {
        knowledgeDbabcdef123456: {
          type: 'http',
          url: 'https://lab.example/api/v1/mcp',
          headers: { Authorization: 'Bearer ${KDBC_MCP_KEY_ABCDEF123456}' },
          tools: ['*'],
        },
      },
    });
    expect(screen.getByText('~/.copilot/mcp-config.json')).toBeInTheDocument();
    const documentation = screen.getByRole('link', { name: 'Client documentation' });
    expect(documentation).toHaveAttribute('target', '_blank');
    expect(documentation).toHaveAttribute('rel', 'noopener noreferrer');
    expect(documentation).toHaveAttribute('href', 'https://code.visualstudio.com/docs/agents/reference/mcp-configuration');

    fireEvent.change(screen.getByLabelText('MCP client'), { target: { value: 'cursor' } });
    const cursor = screen.getByRole('textbox', { name: 'Cursor MCP configuration' }) as HTMLTextAreaElement;
    expect(JSON.parse(cursor.value)).toEqual({
      mcpServers: {
        knowledgeDbabcdef123456: {
          url: 'https://lab.example/api/v1/mcp',
          headers: { Authorization: 'Bearer ${env:KDBC_MCP_KEY_ABCDEF123456}' },
        },
      },
    });
    expect(screen.getByText('.cursor/mcp.json')).toBeInTheDocument();
    expect(documentation).toHaveAttribute('href', 'https://cursor.com/docs/mcp');

    fireEvent.click(screen.getByRole('button', { name: 'Copy configuration' }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(cursor.value));
  } finally {
    if (clipboard) {
      Object.defineProperty(navigator, 'clipboard', clipboard);
    } else {
      Reflect.deleteProperty(navigator, 'clipboard');
    }
  }
});
