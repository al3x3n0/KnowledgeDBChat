import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';

import { apiClient } from '../../../services/api';
import MCPProfileKeyRotation from '../MCPProfileKeyRotation';

jest.mock('../../../services/api', () => ({
  apiClient: {
    getMCPEndpoint: () => 'https://lab.example/api/v1/mcp',
    rotateMCPProfileKey: jest.fn(),
  },
}));

test('confirms rotation, shows the replacement once, and reports the new prefix', async () => {
  jest.mocked(apiClient.rotateMCPProfileKey).mockResolvedValue({
    id: 'profile-1',
    api_key: 'replacement-key',
    key_prefix: 'replacement-',
  });
  const onRotated = jest.fn();
  render(
    <MCPProfileKeyRotation
      profileId="profile-1"
      keyPrefix="original-"
      onRotated={onRotated}
      onRefresh={jest.fn()}
    />
  );
  fireEvent.click(screen.getByRole('button', { name: 'Rotate key' }));
  const dialog = screen.getByRole('dialog');
  expect(dialog).toHaveTextContent('The current key will stop authenticating');
  fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
  expect(apiClient.rotateMCPProfileKey).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Rotate key' }));
  fireEvent.click(screen.getByRole('button', { name: 'Rotate API key' }));
  expect(await screen.findByDisplayValue('replacement-key')).toBeInTheDocument();
  expect(apiClient.rotateMCPProfileKey).toHaveBeenCalledWith('profile-1', 'original-');
  expect(onRotated).toHaveBeenCalledWith('replacement-');
  fireEvent.click(screen.getByRole('button', { name: 'Done' }));
  expect(screen.queryByDisplayValue('replacement-key')).toBeNull();
});
