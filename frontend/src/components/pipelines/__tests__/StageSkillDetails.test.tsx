import { fireEvent, render, screen, within } from '@testing-library/react';
import React, { useState } from 'react';
import { MemoryRouter } from 'react-router-dom';

import type { PipelineVocabulary } from '../../../types';
import type { PipelineStageSpec } from '../PipelineGraph';
import StageInspector from '../StageInspector';

const vocabulary: PipelineVocabulary = {
  evidence_types: [{
    name: 'skill_transform', producers: ['run_sandbox_skill'], typical_seconds: 120,
    job_types: [], perishable: true, consumes: 'an input file',
    sandbox_skill: {
      skill: 'transform', name: 'Transform a dataset', description: 'Convert tabular data into normalized records.',
      produces: 'skill_transform', image: 'data-tool:7', timeout_seconds: 90,
      result_fields: { records: 'number', valid: 'boolean' },
    },
  }],
  job_types: ['research'], tools: [],
};

function Editor({ required = [], catalog = vocabulary }: { required?: string[]; catalog?: PipelineVocabulary | null }) {
  const [stage, setStage] = useState<PipelineStageSpec>({
    id: 'convert', goal: 'Convert input', contract: { required_finding_types: required },
  });
  return <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    <StageInspector stage={stage} allStageIds={['convert']} vocabulary={catalog} onChange={setStage} onDelete={jest.fn()} onClose={jest.fn()} />
    <output data-testid="spec">{JSON.stringify(stage)}</output>
  </MemoryRouter>;
}

it('shows the selected skill image and result contract without setting tool arguments', () => {
  render(<Editor />);
  expect(screen.queryByRole('region', { name: 'Stage Sandbox Skills' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /skill_transform/ }));
  const details = within(screen.getByRole('region', { name: 'Stage Sandbox Skills' }));
  expect(details.getByText('Transform a dataset')).toBeVisible();
  expect(details.getByText('Configured image: data-tool:7')).toBeVisible();
  expect(details.getByText('Command timeout: 90s · Network: disabled')).toBeVisible();
  expect(details.getByText(/goes stale/)).toBeVisible();
  fireEvent.click(details.getByText('Required result.json fields'));
  expect(details.getByText('records')).toBeVisible();
  expect(details.getByText('number')).toBeVisible();
  expect(details.getByText('valid')).toBeVisible();
  expect(details.getByRole('link', { name: 'Edit and validate skills in Tools' })).toHaveAttribute('href', '/tools');
  expect(JSON.parse(screen.getByTestId('spec').textContent || '{}')).toEqual({
    id: 'convert', goal: 'Convert input', contract: { required_finding_types: ['skill_transform'] },
  });
  fireEvent.click(screen.getByRole('button', { name: /skill_transform/ }));
  expect(screen.queryByRole('region', { name: 'Stage Sandbox Skills' })).not.toBeInTheDocument();
});

it.each(['Transform a dataset', 'normalized records', 'data-tool:7'])('finds a skill by %s', (query) => {
  render(<Editor />);
  fireEvent.change(screen.getByLabelText('Filter evidence types'), { target: { value: query } });
  expect(screen.getByRole('button', { name: /skill_transform/ })).toBeVisible();
});

it('explains unavailable selected skill metadata without inventing an image', () => {
  render(<Editor required={['skill_removed']} />);
  const details = within(screen.getByRole('region', { name: 'Stage Sandbox Skills' }));
  expect(details.getByText('skill_removed')).toBeVisible();
  expect(details.getByText(/No active skill details/)).toBeVisible();
  expect(details.queryByText(/Configured image/)).not.toBeInTheDocument();
});

it('distinguishes loading the catalog from an unavailable skill', () => {
  render(<Editor required={['skill_transform']} catalog={null} />);
  expect(screen.getByText('Loading skill details…')).toBeVisible();
  expect(screen.queryByText(/No active skill details/)).not.toBeInTheDocument();
});
