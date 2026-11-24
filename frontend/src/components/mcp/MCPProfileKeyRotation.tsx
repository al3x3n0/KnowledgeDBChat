import React, { useState } from 'react';

import { apiClient } from '../../services/api';
import Button from '../common/Button';
import ConfirmationModal from '../common/ConfirmationModal';
import MCPProfileCredentials from './MCPProfileCredentials';

type State =
  | { status: 'idle' | 'confirm' | 'rotating' }
  | { status: 'complete'; credential: { id: string; api_key: string } };

export default function MCPProfileKeyRotation({ profileId, keyPrefix, onRotated, onRefresh }: {
  profileId: string;
  keyPrefix: string;
  onRotated: (prefix: string) => void;
  onRefresh: () => void;
}) {
  const [state, setState] = useState<State>({ status: 'idle' });
  const [error, setError] = useState('');

  async function rotate() {
    setError('');
    setState({ status: 'rotating' });
    try {
      const credential = await apiClient.rotateMCPProfileKey(profileId, keyPrefix);
      setState({ status: 'complete', credential });
      onRotated(credential.key_prefix);
    } catch {
      setError('Could not confirm rotation. Check the current key before retrying.');
      setState({ status: 'idle' });
      onRefresh();
    }
  }

  return (
    <section className="rounded-md border bg-white p-4">
      {state.status === 'complete' ? (
        <MCPProfileCredentials
          profile={state.credential}
          message="Replacement key ready"
          onDone={() => setState({ status: 'idle' })}
        />
      ) : (
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="font-medium text-gray-900">Profile API key</h3>
            <p className="mt-1 text-sm text-gray-600">
              Generate a replacement key for this profile. The old key will stop authenticating.
            </p>
            {error && <p role="alert" className="mt-2 text-sm text-red-700">{error}</p>}
          </div>
          <Button variant="secondary" onClick={() => setState({ status: 'confirm' })}>Rotate key</Button>
        </div>
      )}
      <ConfirmationModal
        isOpen={state.status === 'confirm' || state.status === 'rotating'}
        isLoading={state.status === 'rotating'}
        onClose={() => setState({ status: 'idle' })}
        onConfirm={() => void rotate()}
        title="Rotate this profile’s API key?"
        message="The current key will stop authenticating as soon as rotation completes. Replace it in each connected client."
        confirmText="Rotate API key"
        variant="danger"
      />
    </section>
  );
}
