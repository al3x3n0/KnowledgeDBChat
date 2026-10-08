/**
 * Directory (LDAP) administration: what is configured, whether it works, and
 * bringing directory users in.
 *
 * LDAP had endpoints and no screen, so setting it up meant editing `.env` and
 * trying to log in -- and a login that fails says nothing about why. The
 * connection test walks the configuration the way a login would (settings,
 * transport, service account, one user's entry) and reports the step that
 * stopped it. It never asks for a user's password.
 *
 * Settings themselves are deployment configuration and are shown, not edited.
 */

import React, { useState } from 'react';
import { useMutation, useQuery } from 'react-query';
import { AlertTriangle, CheckCircle, ShieldCheck, ShieldOff, XCircle } from 'lucide-react';
import toast from 'react-hot-toast';

import type { components } from '../../api/schema';
import { apiClient } from '../../services/api';
import Button from '../common/Button';
import Input from '../common/Input';
import LoadingSpinner from '../common/LoadingSpinner';

type Status = components['schemas']['LdapStatusResponse'];
type TestResult = components['schemas']['LdapTestResponse'];
type ImportResult = components['schemas']['LdapImportResponse'];

const TRANSPORT_LABEL: Record<string, string> = {
  ldaps: 'LDAPS',
  starttls: 'StartTLS',
  plaintext: 'Unencrypted',
};

const Fact: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div>
    <dt className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</dt>
    <dd className="mt-1 text-sm text-gray-900 break-all">{children}</dd>
  </div>
);

const errorText = (error: any, fallback: string): string =>
  error?.response?.data?.detail || error?.message || fallback;

const StatusCard: React.FC<{ status: Status }> = ({ status }) => {
  const problems = status.problems || [];
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex items-center space-x-2">
        {status.enabled ? (
          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-green-100 text-green-800">
            Enabled
          </span>
        ) : (
          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-700">
            Disabled
          </span>
        )}
        {status.enabled && !status.configured && (
          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-yellow-100 text-yellow-800">
            Not fully configured
          </span>
        )}
      </div>

      {!status.enabled ? (
        <p className="text-sm text-gray-600">
          Sign-in uses local accounts only. Set <code>LDAP_ENABLED=true</code> and the
          <code> LDAP_*</code> settings in the backend environment to use a directory.
        </p>
      ) : (
        <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Fact label="Server">{status.uri || '—'}</Fact>
          <Fact label="Base DN">{status.base_dn || '—'}</Fact>
          <Fact label="Transport">
            <span className="inline-flex items-center space-x-1">
              {status.verifies_certificates ? (
                <ShieldCheck className="w-4 h-4 text-green-600" aria-hidden="true" />
              ) : (
                <ShieldOff className="w-4 h-4 text-yellow-600" aria-hidden="true" />
              )}
              <span>
                {TRANSPORT_LABEL[status.transport || 'plaintext'] || status.transport}
                {status.transport !== 'plaintext' &&
                  (status.verifies_certificates
                    ? ', certificate verified'
                    : ', certificate NOT verified')}
              </span>
            </span>
          </Fact>
          <Fact label="Finding users">
            {status.has_service_account ? 'Service account search' : 'User DN template'}
          </Fact>
          <Fact label="Roles from groups">
            {status.role_mapping_configured
              ? `Mapped${status.group_search_configured ? ' (group search)' : ' (memberOf)'}`
              : 'Not mapped — roles are left as they are'}
          </Fact>
          <Fact label="Unknown users">
            {status.create_user_on_login
              ? 'Given an account on first sign-in'
              : 'Refused until imported'}
          </Fact>
          <Fact label="If the directory is unreachable">
            {status.local_fallback_when_unavailable
              ? 'Stored passwords are accepted'
              : 'Directory users cannot sign in'}
          </Fact>
        </dl>
      )}

      {problems.length > 0 && (
        <div className="rounded-md bg-yellow-50 border border-yellow-200 p-4" role="alert">
          <div className="flex items-start space-x-2">
            <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" aria-hidden="true" />
            <div>
              <p className="text-sm font-medium text-yellow-800">
                Directory sign-in will not work with these settings
              </p>
              <ul className="mt-1 list-disc list-inside text-sm text-yellow-800 space-y-1">
                {problems.map((problem) => (
                  <li key={problem}>{problem}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const TestCard: React.FC = () => {
  const [username, setUsername] = useState('');
  const test = useMutation((name: string) => apiClient.testLdap(name.trim() || undefined), {
    onError: (error: any) => {
      toast.error(errorText(error, 'The test could not be run'));
    },
  });
  const result: TestResult | undefined = test.data;

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div>
        <h3 className="text-base font-semibold text-gray-900">Test the connection</h3>
        <p className="text-sm text-gray-600">
          Checks each step a sign-in depends on. Add a login name to see what the directory
          holds for that person — no password is needed or asked for.
        </p>
      </div>
      <form
        className="flex items-end space-x-3"
        onSubmit={(event) => {
          event.preventDefault();
          test.mutate(username);
        }}
      >
        <Input
          label="Login name (optional)"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          placeholder="e.g. jsmith"
          autoComplete="off"
        />
        <Button type="submit" loading={test.isLoading}>
          Run test
        </Button>
      </form>

      {result && (
        <div className="space-y-3">
          <ol className="space-y-2">
            {result.steps.map((step) => (
              <li key={step.step} className="flex items-start space-x-2">
                {step.ok ? (
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" aria-label="passed" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" aria-label="failed" />
                )}
                <span className="text-sm text-gray-900">{step.message}</span>
              </li>
            ))}
          </ol>
          {result.user && (
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4 border-t border-gray-200 pt-3">
              <Fact label="Entry">{result.user.dn}</Fact>
              <Fact label="Account">
                {result.user.username}
                {result.user.email ? ` · ${result.user.email}` : ''}
              </Fact>
              <Fact label="Role">
                {result.user.role || 'Unchanged (no group mapping)'}
              </Fact>
              <Fact label={`Groups (${(result.user.groups || []).length})`}>
                {(result.user.groups || []).length === 0
                  ? 'None found'
                  : (result.user.groups || []).join(' · ')}
              </Fact>
            </dl>
          )}
        </div>
      )}
    </div>
  );
};

const ImportCard: React.FC = () => {
  const [filter, setFilter] = useState('');
  const [preview, setPreview] = useState<ImportResult | null>(null);
  const run = useMutation(
    (dryRun: boolean) =>
      apiClient.importLdapUsers({
        search_filter: filter.trim() || null,
        limit: 200,
        dry_run: dryRun,
        default_role: 'user',
        overwrite_role: false,
      }),
    {
      onSuccess: (data, dryRun) => {
        setPreview(dryRun ? data : null);
        if (!dryRun) {
          toast.success(`Imported: ${data.created} created, ${data.updated} updated`);
        }
      },
      onError: (error: any) => {
        toast.error(errorText(error, 'The import failed'));
      },
    }
  );

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div>
        <h3 className="text-base font-semibold text-gray-900">Import users</h3>
        <p className="text-sm text-gray-600">
          Creates or updates accounts for directory users so they exist before their first
          sign-in. Passwords are never imported. Preview first: nothing changes until you
          apply.
        </p>
      </div>
      <Input
        label="Search filter (optional)"
        value={filter}
        onChange={(event) => {
          setFilter(event.target.value);
          setPreview(null);
        }}
        placeholder="Defaults to LDAP_IMPORT_FILTER"
        autoComplete="off"
        fullWidth
      />
      <div className="flex items-center space-x-3">
        <Button variant="secondary" loading={run.isLoading} onClick={() => run.mutate(true)}>
          Preview
        </Button>
        <Button
          disabled={!preview || preview.created + preview.updated === 0 || run.isLoading}
          onClick={() => run.mutate(false)}
        >
          Apply import
        </Button>
      </div>

      {preview && (
        <div className="space-y-2">
          <p className="text-sm text-gray-900">
            Would create {preview.created}, update {preview.updated}
            {preview.errors > 0 ? `, ${preview.errors} with errors` : ''}.
          </p>
          {preview.rows.length > 0 && (
            <div className="overflow-x-auto border border-gray-200 rounded">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {['User', 'Email', 'Role', 'Action'].map((heading) => (
                      <th
                        key={heading}
                        className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        {heading}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {preview.rows.map((row) => (
                    <tr key={`${row.dn || row.username}`}>
                      <td className="px-4 py-2 text-sm text-gray-900">{row.username}</td>
                      <td className="px-4 py-2 text-sm text-gray-600">{row.email || '—'}</td>
                      <td className="px-4 py-2 text-sm text-gray-600">{row.role}</td>
                      <td className="px-4 py-2 text-sm text-gray-600">
                        {row.action}
                        {row.error ? `: ${row.error}` : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const LdapPanel: React.FC = () => {
  const { data: status, isLoading, error } = useQuery(
    ['ldap-status'],
    () => apiClient.getLdapStatus(),
    { refetchOnWindowFocus: false }
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-900">Directory (LDAP)</h2>
        <p className="text-sm text-gray-600">
          Sign-in against LDAP or Active Directory, and importing its users
        </p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-10">
          <LoadingSpinner size="md" />
        </div>
      ) : error || !status ? (
        <div className="bg-white rounded-lg shadow p-6 text-gray-600">
          The directory status could not be loaded.
        </div>
      ) : (
        <>
          <StatusCard status={status} />
          {status.enabled && <TestCard />}
          {status.enabled && status.configured && (status.problems || []).length === 0 && (
            <ImportCard />
          )}
        </>
      )}
    </div>
  );
};

export default LdapPanel;
