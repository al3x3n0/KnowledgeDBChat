import { fireEvent, render, screen } from '@testing-library/react';
import React, { useState } from 'react';
import { MemoryRouter } from 'react-router-dom';

import type { PipelineTool } from '../../../types';
import StageToolPicker from '../StageToolPicker';

const tools: PipelineTool[] = [
  { name: 'read_file', description: 'Read a working file', source: 'built-in', execution: 'server', job_types: null, produces: [] },
  {
    name: 'p_solver_solve', description: 'Solve constraints', source: 'solver', execution: 'webhook',
    job_types: ['research'], produces: ['solver_result'],
    parameters: {
      type: 'object', required: ['constraints'],
      properties: {
        constraints: { type: 'array', description: 'Constraints to solve', items: { type: 'string' } },
        timeout: { anyOf: [{ type: 'number' }, { type: 'null' }] },
      },
    },
    requires: ['read_file'], consumes: 'a constraint set',
    effects: 'write', network: 'external', cost_tier: 'medium',
    image: 'solver:1', program_images: ['solver-base:1'],
  },
  { name: 'compile_kernel', description: 'Compile a kernel', source: 'uarch_lab', execution: 'server', job_types: ['coding'], produces: ['codegen_measurement'] },
];

function Editor({ initial, jobType = 'research' }: { initial?: string[]; jobType?: string }) {
  const [selected, setSelected] = useState<string[] | undefined>(initial);
  return <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    <StageToolPicker tools={tools} selected={selected} jobType={jobType} onChange={setSelected} onRemoveTool={(name) => setSelected((previous) => previous?.filter((tool) => tool !== name))} />
    <output data-testid="selection">{JSON.stringify(selected ?? null)}</output>
  </MemoryRouter>;
}

it('selects mixed tools and restores the unrestricted menu', () => {
  render(<Editor />);
  fireEvent.change(screen.getByLabelText('Stage tool access'), { target: { value: 'selected' } });
  fireEvent.click(screen.getByLabelText('Allow tool read_file'));
  fireEvent.click(screen.getByLabelText('Allow tool p_solver_solve'));
  expect(screen.getByTestId('selection')).toHaveTextContent('["read_file","p_solver_solve"]');
  expect(screen.queryByLabelText('Allow tool compile_kernel')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Filter stage tools'), { target: { value: 'webhook' } });
  expect(screen.getByLabelText('Allow tool p_solver_solve')).toBeChecked();
  expect(screen.queryByLabelText('Allow tool read_file')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Stage tool access'), { target: { value: 'all' } });
  expect(screen.getByTestId('selection')).toHaveTextContent('null');
});

it('keeps incompatible and removed selections visible so an author can repair them', () => {
  render(<Editor initial={['compile_kernel', 'removed_tool']} />);
  expect(screen.getByText(/Unavailable for research/)).toBeInTheDocument();
  fireEvent.click(screen.getByLabelText('Allow tool compile_kernel'));
  fireEvent.click(screen.getByRole('button', { name: /removed_tool is unavailable/ }));
  expect(screen.getByTestId('selection')).toHaveTextContent('[]');
});

it('allows inspecting tools in the normal menu without changing selections', async () => {
  render(<Editor />);
  expect(screen.getByLabelText('Allow tool p_solver_solve')).toBeChecked();
  expect(screen.getByLabelText('Allow tool p_solver_solve')).toBeDisabled();
  fireEvent.click(screen.getByLabelText('Inspect tool p_solver_solve'));
  expect(await screen.findByText('constraints · array · required')).toBeVisible();
  expect(screen.getByText('timeout · number | null · optional')).toBeVisible();
  expect(screen.getByText('solver_result')).toBeVisible();
  expect(screen.getByText('Prerequisite tools: read_file')).toBeVisible();
  expect(screen.getByText('Configured image: solver:1')).toBeVisible();
  expect(screen.getByText('solver-base:1')).toBeVisible();
  expect(screen.getByText(/Network: external/)).toBeVisible();
  expect(screen.getByTestId('selection')).toHaveTextContent('null');
});

it('handles a catalog with no input metadata and searches by produced evidence', async () => {
  render(<Editor initial={[]} />);
  fireEvent.click(screen.getByLabelText('Inspect tool read_file'));
  expect(await screen.findByText('Input schema unavailable.')).toBeVisible();
  expect(screen.getByText(/No evidence type declared/)).toBeVisible();
  expect(screen.getByTestId('selection')).toHaveTextContent('[]');
  fireEvent.change(screen.getByLabelText('Filter stage tools'), { target: { value: 'solver_result' } });
  expect(screen.getByLabelText('Inspect tool p_solver_solve')).toBeInTheDocument();
  expect(screen.queryByLabelText('Inspect tool read_file')).not.toBeInTheDocument();
});

it('surfaces unavailable argument settings even without an explicit tool selection', () => {
  const remove = jest.fn();
  const select = jest.fn();
  render(<MemoryRouter>
    <StageToolPicker tools={tools} selected={undefined} jobType="research" onChange={select} onRemoveTool={remove}
      defaults={{ missing_defaults: { limit: 0 } }} bindings={{ missing_bindings: { query: { stage: 'source', path: '/value' } } }} />
  </MemoryRouter>);
  expect(screen.getByText('missing_defaults is unavailable')).toBeVisible();
  expect(screen.getByText('missing_bindings is unavailable')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: /missing_bindings is unavailable/ }));
  expect(remove).toHaveBeenCalledWith('missing_bindings');
  expect(select).not.toHaveBeenCalled();
});

it('keeps an incompatible configured tool visible in normal access mode for repair', () => {
  const remove = jest.fn();
  render(<MemoryRouter>
    <StageToolPicker tools={tools} selected={undefined} jobType="research" onChange={jest.fn()} onRemoveTool={remove}
      defaults={{ compile_kernel: { flags: '-O2' } }} />
  </MemoryRouter>);
  expect(screen.getByText(/Unavailable for research/)).toBeVisible();
  expect(screen.getByLabelText('Allow tool compile_kernel')).not.toBeChecked();
  expect(screen.getByLabelText('Allow tool compile_kernel')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Remove compile_kernel and stage settings' }));
  expect(remove).toHaveBeenCalledWith('compile_kernel');
});

it('does not call configured tools unavailable before a catalog is loaded', () => {
  render(<MemoryRouter>
    <StageToolPicker tools={undefined} selected={['unconfirmed']} jobType="research" onChange={jest.fn()} onRemoveTool={jest.fn()} defaults={{ unconfirmed: { limit: 1 } }} />
  </MemoryRouter>);
  expect(screen.getByText(/Tool catalog unavailable/)).toBeVisible();
  expect(screen.queryByText('unconfirmed is unavailable')).not.toBeInTheDocument();
});
