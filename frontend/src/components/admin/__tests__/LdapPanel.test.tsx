import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import LdapPanel from '../LdapPanel';
import { apiClient } from '../../../services/api';

jest.mock('../../../services/api', () => ({
  apiClient: {
    getLdapStatus: jest.fn(),
    testLdap: jest.fn(),
    importLdapUsers: jest.fn(),
  },
}));

const status = apiClient.getLdapStatus as jest.Mock;
const test = apiClient.testLdap as jest.Mock;
const importUsers = apiClient.importLdapUsers as jest.Mock;

const WORKING = {
  enabled: true,
  configured: true,
  uri: 'ldaps://directory.example.com',
  base_dn: 'dc=example,dc=com',
  start_tls: false,
  insecure_skip_tls_verify: false,
  transport: 'ldaps',
  verifies_certificates: true,
  has_service_account: true,
  role_mapping_configured: false,
  group_search_configured: false,
  create_user_on_login: true,
  local_fallback_when_unavailable: false,
  problems: [],
};

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <LdapPanel />
    </QueryClientProvider>
  );
}

describe('LdapPanel', () => {
  beforeEach(() => {
    status.mockReset();
    test.mockReset();
    importUsers.mockReset();
  });

  it('says local accounts only when LDAP is off, and offers nothing to run', async () => {
    status.mockResolvedValue({ ...WORKING, enabled: false, configured: false });
    renderPanel();

    expect(await screen.findByText('Disabled')).toBeInTheDocument();
    expect(screen.getByText(/local accounts only/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run test' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Preview' })).not.toBeInTheDocument();
  });

  it('shows what is wrong with the settings, and withholds the import', async () => {
    status.mockResolvedValue({
      ...WORKING,
      transport: 'plaintext',
      verifies_certificates: false,
      problems: ['The connection is not encrypted: passwords would cross the network in the clear.'],
    });
    renderPanel();

    expect(await screen.findByRole('alert')).toHaveTextContent('not encrypted');
    // The test still runs -- it is how the admin finds out more.
    expect(screen.getByRole('button', { name: 'Run test' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Preview' })).not.toBeInTheDocument();
  });

  it('reports each step of the test and what the directory holds for the user', async () => {
    status.mockResolvedValue(WORKING);
    test.mockResolvedValue({
      ok: true,
      steps: [
        { step: 'service_account', ok: true, message: 'Bound as cn=svc,dc=example,dc=com.' },
        { step: 'user_lookup', ok: true, message: 'Found uid=alice,ou=People,dc=example,dc=com.' },
      ],
      user: {
        username: 'alice',
        dn: 'uid=alice,ou=People,dc=example,dc=com',
        email: 'alice@example.com',
        full_name: 'Alice',
        groups: [],
        role: null,
      },
    });
    renderPanel();

    fireEvent.change(await screen.findByLabelText('Login name (optional)'), {
      target: { value: ' alice ' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run test' }));

    expect(await screen.findByText(/Bound as cn=svc/)).toBeInTheDocument();
    expect(test).toHaveBeenCalledWith('alice');
    expect(screen.getByText(/alice@example.com/)).toBeInTheDocument();
    expect(screen.getByText('Unchanged (no group mapping)')).toBeInTheDocument();
    expect(screen.getAllByLabelText('passed')).toHaveLength(2);
  });

  it('shows the step that failed', async () => {
    status.mockResolvedValue(WORKING);
    test.mockResolvedValue({
      ok: false,
      steps: [
        { step: 'service_account', ok: false, message: 'LDAPSocketOpenError: connection refused' },
      ],
      user: null,
    });
    renderPanel();

    fireEvent.click(await screen.findByRole('button', { name: 'Run test' }));

    expect(await screen.findByText(/connection refused/)).toBeInTheDocument();
    expect(screen.getByLabelText('failed')).toBeInTheDocument();
    expect(test).toHaveBeenCalledWith(undefined);
  });

  it('applies an import only after a preview', async () => {
    status.mockResolvedValue(WORKING);
    importUsers.mockResolvedValue({
      created: 1,
      updated: 0,
      skipped: 0,
      errors: 0,
      rows: [
        { username: 'alice', email: 'alice@example.com', role: 'user', action: 'created', dn: 'uid=alice' },
      ],
    });
    renderPanel();

    const apply = await screen.findByRole('button', { name: 'Apply import' });
    expect(apply).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Preview' }));
    expect(await screen.findByText(/Would create 1, update 0/)).toBeInTheDocument();
    expect(importUsers).toHaveBeenLastCalledWith(expect.objectContaining({ dry_run: true }));

    await waitFor(() => expect(apply).not.toBeDisabled());
    fireEvent.click(apply);
    await waitFor(() =>
      expect(importUsers).toHaveBeenLastCalledWith(expect.objectContaining({ dry_run: false }))
    );
  });
});
