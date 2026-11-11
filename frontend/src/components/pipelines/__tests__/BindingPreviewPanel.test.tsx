import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import BindingPreviewPanel from '../BindingPreviewPanel';
import type { PipelineBindingPreview } from '../../../types';

jest.mock('../../../services/api', () => ({ apiClient: { previewPipelineBindings: jest.fn() } }));
const { apiClient } = jest.requireMock('../../../services/api');
const stage = { id: 'consume', depends_on: ['produce'], tool_bindings: { check: { code: { stage: 'produce', path: '/value' } } } };
const spec = { name: 'p', stages: [stage] };
const response: PipelineBindingPreview = {
  root_job_id: 'root', stage: 'consume', omitted: 2,
  items: [{ tool: 'check', argument: 'code', source_stage: 'produce', path: '/value', source_job_id: 'source-job', source_status: 'completed', source_contract_satisfied: false, status: 'resolved', value_type: 'string', value_json: '"result"', truncated: true }],
};
beforeEach(() => jest.clearAllMocks());

it('previews recorded values with provenance and explicit limits on request', async () => {
  apiClient.previewPipelineBindings.mockResolvedValue(response);
  const open = jest.fn();
  render(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} onOpenJob={open} />);
  expect(apiClient.previewPipelineBindings).not.toHaveBeenCalled();
  fireEvent.click(screen.getByText('Preview applied bindings'));
  expect(await screen.findByText('"result"')).toBeInTheDocument();
  expect(apiClient.previewPipelineBindings).toHaveBeenCalledWith('root', spec, 'consume');
  expect(screen.getByText(/contract unmet/)).toBeInTheDocument();
  expect(screen.getByText('Value preview truncated.')).toBeInTheDocument();
  expect(screen.getByText(/2 additional bindings omitted/)).toBeInTheDocument();
  fireEvent.click(screen.getByText('Open source job'));
  expect(open).toHaveBeenCalledWith('source-job');
});

it('does not preview without a recorded run', () => {
  render(<BindingPreviewPanel stage={stage} spec={spec} />);
  expect(screen.getByText('Preview applied bindings')).toBeDisabled();
  expect(apiClient.previewPipelineBindings).not.toHaveBeenCalled();
});

it('discards stale values when the edited spec changes during a request', async () => {
  let finish!: (value: PipelineBindingPreview) => void;
  apiClient.previewPipelineBindings.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const view = render(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} />);
  fireEvent.click(screen.getByText('Preview applied bindings'));
  await waitFor(() => expect(apiClient.previewPipelineBindings).toHaveBeenCalled());
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={{ ...spec, name: 'edited' }} />);
  await act(async () => { finish(response); });
  expect(screen.queryByText('"result"')).not.toBeInTheDocument();
  expect(screen.getByText('Preview applied bindings')).not.toBeDisabled();
});

it('shows resolution errors and request failures', async () => {
  apiClient.previewPipelineBindings.mockResolvedValue({ ...response, items: [{ ...response.items[0], status: 'invalid', error: 'Result path missing' }] });
  render(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} />);
  fireEvent.click(screen.getByText('Preview applied bindings'));
  expect(await screen.findByText('Result path missing')).toBeInTheDocument();
  apiClient.previewPipelineBindings.mockRejectedValue({ response: { data: { detail: 'Pipeline run not found' } } });
  fireEvent.click(screen.getByText('Preview applied bindings'));
  expect(await screen.findByRole('alert')).toHaveTextContent('Pipeline run not found');
});

it('invalidates a displayed preview on catalog refresh and requires another request', async () => {
  apiClient.previewPipelineBindings.mockResolvedValue(response);
  const view = render(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} catalogRevision={1} />);
  fireEvent.click(screen.getByText('Preview applied bindings'));
  await screen.findByText('"result"');
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} catalogRevision={1} catalogRefreshing />);
  expect(screen.queryByText('"result"')).not.toBeInTheDocument();
  expect(screen.getByText('Preview applied bindings')).toBeDisabled();
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} catalogRevision={2} />);
  expect(screen.queryByText('"result"')).not.toBeInTheDocument();
  expect(screen.getByText('Preview applied bindings')).toBeEnabled();
  expect(apiClient.previewPipelineBindings).toHaveBeenCalledTimes(1);
  apiClient.previewPipelineBindings.mockResolvedValue({ ...response, items: [{ ...response.items[0], status: 'invalid', error: 'Input schema changed' }] });
  fireEvent.click(screen.getByText('Preview applied bindings'));
  expect(await screen.findByText('Input schema changed')).toBeInTheDocument();
});

it.each(['result', 'error'])('ignores a preview %s that arrives after the catalog changes', async (outcome) => {
  let resolve!: (value: PipelineBindingPreview) => void;
  let reject!: (error: unknown) => void;
  apiClient.previewPipelineBindings.mockImplementationOnce(() => new Promise((done, fail) => { resolve = done; reject = fail; }));
  const view = render(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} catalogRevision={1} />);
  fireEvent.click(screen.getByText('Preview applied bindings'));
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} catalogRevision={1} catalogRefreshing />);
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} catalogRevision={2} />);
  await act(async () => {
    if (outcome === 'result') resolve(response);
    else reject({ response: { data: { detail: 'old catalog error' } } });
  });
  expect(screen.queryByText('"result"')).not.toBeInTheDocument();
  expect(screen.queryByText('old catalog error')).not.toBeInTheDocument();
  expect(screen.getByText('Preview applied bindings')).toBeEnabled();
});

it('does not resurrect an old preview when an edited spec is restored', async () => {
  apiClient.previewPipelineBindings.mockResolvedValue(response);
  const view = render(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} />);
  fireEvent.click(screen.getByText('Preview applied bindings'));
  await screen.findByText('"result"');
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={{ ...spec, name: 'edited' }} />);
  view.rerender(<BindingPreviewPanel rootJobId="root" stage={stage} spec={spec} />);
  expect(screen.queryByText('"result"')).not.toBeInTheDocument();
  expect(apiClient.previewPipelineBindings).toHaveBeenCalledTimes(1);
});
