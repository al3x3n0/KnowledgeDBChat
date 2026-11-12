/**
 * The pipeline editor, and the distinctions it has to keep separate.
 *
 * Its job is to be the place a bad pipeline dies, so the assertions here are
 * mostly about failure being visible and specific: unparseable text is not the
 * same as an invalid pipeline, and an invalid pipeline is not the same as an
 * unaffordable one.
 */

import { act, fireEvent, render as rtlRender, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';

import PipelineStudioPage from '../PipelineStudioPage';

// The page now shows the last run of whatever pipeline is open, and that
// section fetches. Wrapping here rather than at twenty call sites.
const render = (ui: React.ReactElement) =>
  rtlRender(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{ui}</MemoryRouter>
    </QueryClientProvider>
  );

jest.mock('../../services/api', () => ({
  apiClient: {
    checkPipeline: jest.fn(),
    bindPipeline: jest.fn(),
    launchPipeline: jest.fn(),
    // The library loads on mount. Unmocked it throws into a deliberately
    // silent catch — the tests pass while the page errors on every render,
    // which is worse than a failure because nothing says so.
    listSavedPipelines: jest.fn(),
    // Same reason: the chain-import survey also loads on mount into a silent
    // catch.
    surveyChainsForImport: jest.fn(),
    suggestContracts: jest.fn(),
    importChainAsPipeline: jest.fn(),
    saveSavedPipeline: jest.fn(),
    updateSavedPipeline: jest.fn(),
    deleteSavedPipeline: jest.fn(),
    getPipelineRun: jest.fn(),
    // The stage inspector asks for the evidence vocabulary on mount. Left out
    // of this mock it is not merely unstubbed but *undefined*, so the call
    // throws a TypeError synchronously -- before any promise exists for the
    // page's deliberate `.catch(() => {})` to absorb. Every test in the file
    // died on render, which is exactly the shape the note above warns about,
    // arriving from the other direction.
    getPipelineVocabulary: jest.fn(),
    previewPipelineBindings: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

const apiClient = require('../../services/api').apiClient;

// Enough of the real vocabulary for the inspector to render its pickers. The
// editor is deliberately usable without it, so no test here depends on the
// contents -- only on the call not exploding.
const vocabulary = {
  evidence_types: [
    {
      name: 'benchmark_measurement',
      producers: ['benchmark_c_snippet'],
      typical_seconds: 120,
      job_types: ['analysis', 'coding'],
      perishable: true,
    },
    {
      name: 'document',
      producers: ['search_documents'],
      typical_seconds: 20,
      job_types: [],
      perishable: false,
    },
  ],
  job_types: ['research', 'analysis', 'coding'],
  tools: [
    { name: 'p_data_fetch', description: 'Fetch a dataset', source: 'data', execution: 'webhook', job_types: ['research'], produces: [], parameters: { type: 'object', properties: { query: { type: 'string' } } } },
  ],
};

const okCheck = {
  valid: true,
  problems: [],
  expressible: true,
  binding_problems: [],
  description: ['study:', '  2 stages'],
  plan: {
    order: ['profile', 'measure'],
    stages: [
      {
        stage_id: 'profile',
        tools: ['profile_c_workload'],
        iterations: 1,
        seconds: 60,
        checkpoint: false,
        unpriced: [],
      },
      {
        stage_id: 'measure',
        tools: ['benchmark_c_snippet'],
        iterations: 1,
        seconds: 120,
        checkpoint: true,
        unpriced: [],
      },
    ],
    total_seconds: 180,
    critical_path_seconds: 180,
    checkpoints: ['measure'],
  },
  budget: null,
};

beforeEach(() => {
  jest.clearAllMocks();
  window.localStorage.clear();
  apiClient.checkPipeline.mockResolvedValue(okCheck);
  apiClient.listSavedPipelines.mockResolvedValue([]);
  // A pipeline with a last run fetches it. Most of these tests are not about
  // the run, and a query function returning undefined is an error rather than
  // an empty answer.
  apiClient.getPipelineRun.mockRejectedValue({ response: { status: 404 } });
  apiClient.getPipelineVocabulary.mockResolvedValue(vocabulary);
  apiClient.surveyChainsForImport.mockResolvedValue({ candidates: [] });
  // Follows the editor on a debounce; unmocked it throws into a silent catch.
  apiClient.suggestContracts.mockResolvedValue({ suggestions: {} });
});

const typeSpec = (text: string) =>
  fireEvent.change(screen.getByLabelText('Pipeline specification'), {
    target: { value: text },
  });

describe('refreshing the tool catalog', () => {
  const configured = {
    name: 'data', stages: [
      { id: 'profile', goal: 'Find data', contract: { required_finding_types: ['document'] } },
      { id: 'measure', goal: 'Convert data', depends_on: ['profile'], tools: ['p_data_fetch'],
        contract: { required_finding_types: ['skill_convert'] },
        tool_defaults: { p_data_fetch: { query: 'saved query' } },
        tool_bindings: { p_data_fetch: { query: { stage: 'profile', path: '/query' } } },
      },
    ],
  };
  const sourceText = () => (screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value;
  const refresh = () => screen.getByRole('button', { name: 'Refresh tools and skills' });
  const launch = () => screen.getByRole('button', { name: /Launch/i });

  it('refreshes skill details without replacing the spec, selection, or unapplied argument edits', async () => {
    render(<PipelineStudioPage />);
    await waitFor(() => expect(launch()).toBeEnabled());
    typeSpec(JSON.stringify(configured));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="measure"]')!);
    fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
    const argumentEditor = await screen.findByLabelText('Argument defaults for p_data_fetch');
    fireEvent.change(argumentEditor, { target: { value: '{"query":"unsaved edit"}' } });
    await waitFor(() => expect(launch()).toBeEnabled());
    const original = sourceText();
    const checksBefore = apiClient.checkPipeline.mock.calls.length;
    apiClient.getPipelineVocabulary.mockResolvedValue({ ...vocabulary, evidence_types: [
      ...vocabulary.evidence_types,
      { name: 'skill_convert', producers: ['run_sandbox_skill'], typical_seconds: 120,
        job_types: [], perishable: true, consumes: 'a dataset', sandbox_skill: {
          skill: 'convert', name: 'Convert data', description: 'Convert a dataset to normalized records.',
          produces: 'skill_convert', image: 'dataset-tools:8', timeout_seconds: 90,
          result_fields: { records: 'number' },
        } },
    ] });
    fireEvent.click(refresh());
    expect(launch()).toBeDisabled();
    expect(await screen.findByText('Configured image: dataset-tools:8')).toBeVisible();
    expect(screen.getByLabelText('Stage id')).toHaveValue('measure');
    expect(screen.getByLabelText('Argument defaults for p_data_fetch')).toHaveValue('{"query":"unsaved edit"}');
    expect(sourceText()).toBe(original);
    await waitFor(() => expect(apiClient.checkPipeline.mock.calls.length).toBeGreaterThan(checksBefore));
    expect(apiClient.checkPipeline).toHaveBeenLastCalledWith(configured, undefined);
    await waitFor(() => expect(launch()).toBeEnabled());
  });

  it('keeps removed tool settings for repair and rechecks them before enabling Launch', async () => {
    render(<PipelineStudioPage />);
    await waitFor(() => expect(launch()).toBeEnabled());
    typeSpec(JSON.stringify(configured));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="measure"]')!);
    await waitFor(() => expect(launch()).toBeEnabled());
    apiClient.getPipelineVocabulary.mockResolvedValue({ ...vocabulary, tools: [] });
    apiClient.checkPipeline.mockResolvedValue({ ...okCheck, valid: false, plan: null,
      problems: ['measure: tool p_data_fetch is unavailable'],
    });
    fireEvent.click(refresh());
    expect(launch()).toBeDisabled();
    expect(await screen.findByRole('button', { name: /p_data_fetch is unavailable/ })).toBeVisible();
    await screen.findByText('measure: tool p_data_fetch is unavailable');
    expect(launch()).toBeDisabled();
    expect(JSON.parse(sourceText())).toEqual(configured);
    fireEvent.click(screen.getByRole('button', { name: /p_data_fetch is unavailable/ }));
    const repaired = JSON.parse(sourceText());
    expect(repaired.stages[1].tools).toEqual([]);
    expect(repaired.stages[1].tool_defaults).toBeUndefined();
    expect(repaired.stages[1].tool_bindings).toBeUndefined();
    expect(repaired.stages[0]).toEqual(configured.stages[0]);
    await waitFor(() => expect(apiClient.checkPipeline).toHaveBeenLastCalledWith(repaired, undefined));
  });

  it('retries an initial failure and keeps the last catalog after a later failure', async () => {
    apiClient.getPipelineVocabulary.mockRejectedValueOnce(new Error('unavailable'));
    render(<PipelineStudioPage />);
    expect(await screen.findByText(/Could not refresh the tool catalog/)).toBeVisible();
    typeSpec(JSON.stringify(configured));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="measure"]')!);
    expect(screen.getByText('Tool catalog unavailable. Refresh tools and skills to retry.')).toBeVisible();
    const original = sourceText();
    fireEvent.click(refresh());
    expect(await screen.findByLabelText('Allow tool p_data_fetch')).toBeVisible();
    expect(screen.queryByText(/Could not refresh the tool catalog/)).not.toBeInTheDocument();
    await waitFor(() => expect(refresh()).toBeEnabled());
    apiClient.getPipelineVocabulary.mockRejectedValueOnce(new Error('unavailable again'));
    fireEvent.click(refresh());
    expect(await screen.findByText('Showing the last loaded catalog.')).toBeVisible();
    expect(screen.getByLabelText('Allow tool p_data_fetch')).toBeVisible();
    expect(sourceText()).toBe(original);
    expect(screen.getByLabelText('Stage id')).toHaveValue('measure');
    expect(apiClient.getPipelineVocabulary).toHaveBeenCalledTimes(3);
  });
});

it('clears unavailable configuration in normal access mode without removing another tool settings', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  const reference = { stage: 'produce', path: '/value' };
  const original = { stages: [{ id: 'produce' }, { id: 'consume', tools: null, depends_on: ['produce'],
    tool_defaults: { p_removed: { query: 'remove' }, p_data_fetch: { query: 'keep' } },
    tool_bindings: { p_removed: { query: reference }, p_data_fetch: { query: reference } },
  }] };
  typeSpec(JSON.stringify(original));
  fireEvent.click(document.querySelector('.react-flow__node[data-id="consume"]')!);
  expect(screen.getByText('p_removed is unavailable')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: /p_removed is unavailable/ }));
  const next = JSON.parse((screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value).stages[1];
  expect(next.tools).toBeNull();
  expect(next.tool_defaults).toEqual({ p_data_fetch: { query: 'keep' } });
  expect(next.tool_bindings).toEqual({ p_data_fetch: { query: reference } });
  expect(next.depends_on).toEqual(['produce']);
  expect(screen.queryByText('p_removed is unavailable')).not.toBeInTheDocument();
});

it('keeps bindings visible for explicit repair after deleting their source stage', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  const original = { name: 'p', stages: [
    { id: 'produce', goal: 'Produce' }, { id: 'other', goal: 'Other source' },
    { id: 'consume', goal: 'Consume', depends_on: ['produce', 'other'], may_revisit: ['produce', 'other'],
      tools: ['p_data_fetch'], tool_defaults: { p_data_fetch: { query: 'a default' } },
      tool_bindings: { p_data_fetch: { query: { stage: 'produce', path: '/query' } } },
    },
  ] };
  typeSpec(JSON.stringify(original));
  fireEvent.click(document.querySelector('.react-flow__node[data-id="produce"]')!);
  fireEvent.click(screen.getByLabelText('Delete stage produce'));
  const current = () => JSON.parse((screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value);
  expect(current().stages[1]).toMatchObject({ depends_on: ['other'], may_revisit: ['other'], tool_bindings: original.stages[2].tool_bindings });
  fireEvent.click(document.querySelector('.react-flow__node[data-id="consume"]')!);
  expect(screen.getByRole('region', { name: 'Stage binding issues' })).toHaveTextContent("source stage 'produce' no longer exists");
  fireEvent.click(screen.getByLabelText('Inspect tool p_data_fetch'));
  fireEvent.change(await screen.findByLabelText('Binding source 1 for p_data_fetch'), { target: { value: 'other' } });
  expect(current().stages[1].tool_bindings.p_data_fetch.query.stage).toBe('produce');
  fireEvent.click(screen.getByText('Apply bindings'));
  expect(current().stages[1].tool_bindings.p_data_fetch.query).toEqual({ stage: 'other', path: '/query' });
  expect(current().stages[1].tool_defaults).toEqual(original.stages[2].tool_defaults);
  expect(screen.queryByRole('region', { name: 'Stage binding issues' })).not.toBeInTheDocument();
});

it('writes stage tool selections into the checked specification', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  const node = document.querySelector('.react-flow__node[data-id="profile"]');
  expect(node).not.toBeNull();
  fireEvent.click(node!);
  fireEvent.change(await screen.findByLabelText('Stage tool access'), { target: { value: 'selected' } });
  fireEvent.click(screen.getByLabelText('Allow tool p_data_fetch'));
  const editor = screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement;
  expect(JSON.parse(editor.value).stages[0].tools).toEqual(['p_data_fetch']);
  await waitFor(() => {
    const calls = apiClient.checkPipeline.mock.calls;
    expect(calls[calls.length - 1][0].stages[0].tools).toEqual(['p_data_fetch']);
  });
});

it('applies tool defaults to the stage and attaches the tool in restricted mode', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  fireEvent.click(document.querySelector('.react-flow__node[data-id="profile"]')!);
  fireEvent.change(await screen.findByLabelText('Stage tool access'), { target: { value: 'selected' } });
  fireEvent.click(screen.getByLabelText('Inspect tool p_data_fetch'));
  const input = await screen.findByLabelText('Argument defaults for p_data_fetch');
  fireEvent.change(input, { target: { value: '{"limit":5}' } });
  fireEvent.click(screen.getByText('Apply defaults'));
  const editor = screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement;
  const stage = JSON.parse(editor.value).stages[0];
  expect(stage.tools).toEqual(['p_data_fetch']);
  expect(stage.tool_defaults).toEqual({ p_data_fetch: { limit: 5 } });
});

it('does not carry an unapplied defaults edit into another stage', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  fireEvent.click(document.querySelector('.react-flow__node[data-id="profile"]')!);
  fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
  fireEvent.change(await screen.findByLabelText('Argument defaults for p_data_fetch'), { target: { value: '{"limit":5}' } });
  fireEvent.click(document.querySelector('.react-flow__node[data-id="measure"]')!);
  fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
  expect(await screen.findByLabelText('Argument defaults for p_data_fetch')).toHaveValue('{}');
});

it('writes dependency argument bindings into the stage specification', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  fireEvent.click(document.querySelector('.react-flow__node[data-id="measure"]')!);
  fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
  fireEvent.click(await screen.findByText('Add binding'));
  fireEvent.change(screen.getByLabelText('Binding path 1 for p_data_fetch'), { target: { value: '/structured_output/query' } });
  fireEvent.click(screen.getByText('Apply bindings'));
  const source = (screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value;
  expect(JSON.parse(source).stages[1].tool_bindings).toEqual({ p_data_fetch: { query: { stage: 'profile', path: '/structured_output/query' } } });
  expect(screen.getByRole('region', { name: 'Binding preview' })).toBeInTheDocument();
  expect(screen.getByText('Preview applied bindings')).toBeDisabled();
});

it('commits a stage rename with its dependencies and bindings, and refuses collisions', async () => {
  render(<PipelineStudioPage />);
  await screen.findAllByText('profile_c_workload');
  typeSpec(JSON.stringify({ name: 'p', stages: [
    { id: 'produce', goal: 'Produce' },
    { id: 'consume', goal: 'Consume', depends_on: ['produce'], tool_bindings: { p_data_fetch: { query: { stage: 'produce', path: '/query' } } } },
  ] }));
  fireEvent.click(document.querySelector('.react-flow__node[data-id="produce"]')!);
  const id = await screen.findByLabelText('Stage id');
  fireEvent.change(id, { target: { value: 'new_name' } });
  fireEvent.blur(id);
  const current = () => JSON.parse((screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value);
  expect(current().stages[0].id).toBe('new_name');
  expect(current().stages[1].depends_on).toEqual(['new_name']);
  expect(current().stages[1].tool_bindings.p_data_fetch.query.stage).toBe('new_name');
  fireEvent.change(screen.getByLabelText('Stage id'), { target: { value: 'consume' } });
  fireEvent.blur(screen.getByLabelText('Stage id'));
  expect(screen.getByRole('alert')).toHaveTextContent('Rename not applied');
  expect(current().stages[0].id).toBe('new_name');
});

it('shows the tools it derived, which the author never writes', async () => {
  render(<PipelineStudioPage />);

  // The point of contracts: you say what must be true, and the tools that get
  // there are deduced. An author who names tools can name an impossible one.
  //
  // getAllBy, because the default split view shows each tool twice — once in
  // the plan and once on its node in the graph. Two views of one document is
  // the design, so a query that demanded uniqueness would be asserting the
  // opposite of what this page is for.
  expect((await screen.findAllByText('profile_c_workload')).length).toBeGreaterThan(0);
  expect(screen.getAllByText('benchmark_c_snippet').length).toBeGreaterThan(0);
});

it('reports a checkpoint before the run stops at it, not when it does', async () => {
  render(<PipelineStudioPage />);

  expect(await screen.findByText(/stops at measure/i)).toBeInTheDocument();
  expect(screen.getByText('checkpoint')).toBeInTheDocument();
});

it('separates unparseable text from an invalid pipeline', async () => {
  render(<PipelineStudioPage />);
  // findAll, like every other wait on this text in this file: the tool appears
  // once per stage that lists it, so the singular form asserts a render order
  // rather than that the page rendered. It passed only while this file happened
  // to run early enough to catch a single stage, and any edit that perturbed
  // jest's file ordering broke it.
  await screen.findAllByText('profile_c_workload');

  typeSpec('{ not json');

  expect(await screen.findByText('Not valid JSON')).toBeInTheDocument();
  // And it does not ask the server about text that is not a document — that
  // would only return a less specific version of the same complaint.
  const callsBefore = apiClient.checkPipeline.mock.calls.length;
  await new Promise((resolve) => setTimeout(resolve, 700));
  expect(apiClient.checkPipeline.mock.calls.length).toBe(callsBefore);
});

it('lists every problem, not just the first', async () => {
  apiClient.checkPipeline.mockResolvedValue({
    ...okCheck,
    valid: false,
    plan: null,
    problems: [
      "attribute: assumes 'counter_trace', which no stage before it produces",
      'b: no tool produces telepathy -- the contract cannot be satisfied',
    ],
  });
  render(<PipelineStudioPage />);

  expect(await screen.findByText('2 problems')).toBeInTheDocument();
  expect(screen.getByText(/counter_trace/)).toBeInTheDocument();
  expect(screen.getByText(/telepathy/)).toBeInTheDocument();
});

it('says a valid pipeline is unaffordable without calling it invalid', async () => {
  // Three separate answers. A well-formed pipeline that costs too much is a
  // different problem from a broken one, and collapsing them would send the
  // author to edit the wrong thing.
  apiClient.checkPipeline.mockResolvedValue({
    ...okCheck,
    budget: {
      affordable: false,
      budget_seconds: 60,
      estimated_seconds: 1200,
      critical_path_seconds: 1200,
      unpriced_tools: [],
    },
  });
  render(<PipelineStudioPage />);

  expect(await screen.findByText('Spec valid')).toBeInTheDocument();
  expect(screen.getByText(/Needs 20 min/)).toBeInTheDocument();
});

it('surfaces the server saying it is not a pipeline at all', async () => {
  apiClient.checkPipeline.mockRejectedValue({
    response: { data: { detail: "Not a pipeline spec: 'stages' must be a list, got str" } },
  });
  render(<PipelineStudioPage />);

  expect(await screen.findByText('Not a pipeline')).toBeInTheDocument();
  expect(screen.getByText(/'stages' must be a list/)).toBeInTheDocument();
});

it('marks unpriced tools rather than letting a stage look free', async () => {
  apiClient.checkPipeline.mockResolvedValue({
    ...okCheck,
    plan: {
      ...okCheck.plan,
      stages: [
        {
          stage_id: 'profile',
          tools: ['mystery_tool'],
          iterations: 1,
          seconds: 0,
          checkpoint: false,
          unpriced: ['mystery_tool'],
        },
      ],
      order: ['profile'],
      total_seconds: 0,
    },
  });
  render(<PipelineStudioPage />);

  // Zero seconds because nothing knows the cost is not the same as free.
  expect(await screen.findByText('1 unpriced')).toBeInTheDocument();
});


describe('launching', () => {
  it('requires a fresh check immediately after changing stage configuration', async () => {
    render(<PipelineStudioPage />);
    const launch = screen.getByRole('button', { name: /Launch/i });
    await waitFor(() => expect(launch).toBeEnabled());
    apiClient.checkPipeline.mockImplementation(() => new Promise(() => {}));
    const previous = JSON.parse((screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value);
    previous.stages[0].tool_defaults = { search_documents: { query: 'a different input' } };
    typeSpec(JSON.stringify(previous));
    expect(launch).toBeDisabled();
    fireEvent.click(launch);
    expect(apiClient.launchPipeline).not.toHaveBeenCalled();
    expect(screen.queryByText('Spec valid')).not.toBeInTheDocument();
    await waitFor(() => expect(apiClient.checkPipeline).toHaveBeenLastCalledWith(previous, undefined));
    expect(launch).toBeDisabled();
  });

  it('invalidates the old verdict immediately when the budget changes', async () => {
    render(<PipelineStudioPage />);
    const launch = screen.getByRole('button', { name: /Launch/i });
    await waitFor(() => expect(launch).toBeEnabled());
    apiClient.checkPipeline.mockResolvedValue({ ...okCheck, budget: {
      affordable: false, budget_seconds: 1, estimated_seconds: 180,
      critical_path_seconds: 180, unpriced_tools: [],
    } });
    fireEvent.change(screen.getByLabelText('Budget (seconds)'), { target: { value: '1' } });
    expect(launch).toBeDisabled();
    await screen.findByText(/Needs 3 min/);
    expect(launch).toBeDisabled();
  });

  it.each(['result', 'error'])('ignores an older check %s after the current check finishes', async (outcome) => {
    let resolveOld!: (value: unknown) => void;
    let rejectOld!: (error: unknown) => void;
    apiClient.checkPipeline.mockImplementationOnce(() => new Promise((resolve, reject) => {
      resolveOld = resolve; rejectOld = reject;
    }));
    render(<PipelineStudioPage />);
    await waitFor(() => expect(apiClient.checkPipeline).toHaveBeenCalledTimes(1));
    const next = { name: 'newer', stages: [{ id: 'newer', goal: 'Current work' }] };
    typeSpec(JSON.stringify(next));
    const launch = screen.getByRole('button', { name: /Launch/i });
    await waitFor(() => expect(apiClient.checkPipeline).toHaveBeenCalledWith(next, undefined));
    await waitFor(() => expect(launch).toBeEnabled());
    await act(async () => {
      if (outcome === 'result') resolveOld({ ...okCheck, valid: false, plan: null, problems: ['older spec failed'] });
      else rejectOld({ response: { data: { detail: 'older check unavailable' } } });
    });
    expect(launch).toBeEnabled();
    expect(screen.queryByText('older spec failed')).not.toBeInTheDocument();
    expect(screen.queryByText('older check unavailable')).not.toBeInTheDocument();
  });

  it('does not accept an in-flight success for text edited into invalid JSON', async () => {
    let resolveOld!: (value: unknown) => void;
    apiClient.checkPipeline.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }));
    render(<PipelineStudioPage />);
    await waitFor(() => expect(apiClient.checkPipeline).toHaveBeenCalledTimes(1));
    typeSpec('{ invalid');
    await act(async () => { resolveOld(okCheck); });
    expect(screen.getByRole('button', { name: /Launch/i })).toBeDisabled();
    expect(screen.queryByText('Spec valid')).not.toBeInTheDocument();
    await screen.findByText('Not valid JSON');
  });

  it('does not offer to launch a pipeline the server would refuse', async () => {
    apiClient.checkPipeline.mockResolvedValue({
      ...okCheck,
      valid: false,
      plan: null,
      problems: ['a: no tool produces telepathy'],
    });
    render(<PipelineStudioPage />);
    await screen.findByText('1 problem');

    // Offering a button whose only outcome is a refusal teaches the user that
    // buttons here do not mean anything.
    expect(screen.getByRole('button', { name: /Launch/i })).toBeDisabled();
  });

  it('does not offer to launch a pipeline that cannot afford itself', async () => {
    apiClient.checkPipeline.mockResolvedValue({
      ...okCheck,
      budget: {
        affordable: false,
        budget_seconds: 60,
        estimated_seconds: 1200,
        critical_path_seconds: 1200,
        unpriced_tools: [],
      },
    });
    render(<PipelineStudioPage />);
    await screen.findByText(/Needs 20 min/);

    expect(screen.getByRole('button', { name: /Launch/i })).toBeDisabled();
  });

  it('names the price and the stops before spending anything', async () => {
    const confirm = jest.spyOn(window, 'confirm').mockReturnValue(false);
    render(<PipelineStudioPage />);
    await screen.findAllByText('profile_c_workload');

    fireEvent.click(screen.getByRole('button', { name: /Launch/i }));

    const asked = confirm.mock.calls[0][0] as string;
    expect(asked).toContain('3 min');
    expect(asked).toContain('measure'); // the checkpoint it will stop at
    // Declining means nothing was started.
    expect(apiClient.launchPipeline).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it('sends the estimate it showed, so an edited spec is refused', async () => {
    jest.spyOn(window, 'confirm').mockReturnValue(true);
    apiClient.launchPipeline.mockResolvedValue({
      job_id: 'job-9',
      name: 'study',
      stages: ['profile', 'measure'],
      estimated_seconds: 180,
      checkpoints: [],
    });
    render(<PipelineStudioPage />);
    await screen.findAllByText('profile_c_workload');

    fireEvent.click(screen.getByRole('button', { name: /Launch/i }));

    await waitFor(() =>
      expect(apiClient.launchPipeline).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({ acknowledgedSeconds: 180 })
      )
    );
    // And it goes to the run it just started, rather than leaving the user on
    // an editor with no sign anything happened.
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith('/autonomous-agents?job=job-9'));
  });
});


describe('the library', () => {
  const stored = {
    id: 'p-1',
    name: 'Attention survey',
    spec: { name: 'attention-survey', stages: [{ id: 'gather', contract: {} }] },
    last_check_valid: 'valid' as const,
    last_estimated_seconds: 540,
    launch_count: 3,
    last_job_id: 'job-7',
  };

  const withControls = {
    name: 'portable', stages: [
      { id: 'produce', goal: 'Produce' },
      { id: 'consume', goal: 'Consume', depends_on: ['produce'], tools: ['p_data_fetch'],
        tool_defaults: { p_data_fetch: { query: 'saved query' } },
        tool_bindings: { p_data_fetch: { query: { stage: 'produce', path: '/query' } } },
      },
    ],
  };

  it('discards unapplied tool edits when opening a pipeline with the same stage ids', async () => {
    apiClient.listSavedPipelines.mockResolvedValue([
      { ...stored, spec: withControls, last_job_id: null },
      { ...stored, id: 'p-2', name: 'Another survey', spec: withControls, last_job_id: null },
    ]);
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="consume"]')!);
    fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
    fireEvent.change(await screen.findByLabelText('Argument defaults for p_data_fetch'), {
      target: { value: '{"query":"unapplied defaults"}' },
    });
    fireEvent.change(screen.getByLabelText('Binding path 1 for p_data_fetch'), {
      target: { value: '/unapplied' },
    });

    fireEvent.click(screen.getByText('Another survey'));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="consume"]')!);
    if (!screen.queryByLabelText('Argument defaults for p_data_fetch')) {
      fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
    }
    expect(await screen.findByLabelText('Argument defaults for p_data_fetch')).toHaveValue(
      JSON.stringify({ query: 'saved query' }, null, 2)
    );
    expect(screen.getByLabelText('Binding path 1 for p_data_fetch')).toHaveValue('/query');
    expect(JSON.parse((screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value)).toEqual(withControls);
  });

  it('saves edited defaults and reopens them with the original tool selection and bindings', async () => {
    let record = { ...stored, spec: withControls };
    apiClient.listSavedPipelines.mockImplementation(async () => [record]);
    apiClient.updateSavedPipeline.mockImplementationOnce(async (_id: string, update: { spec: typeof withControls }) => {
      record = { ...record, spec: update.spec };
      return record;
    });
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="consume"]')!);
    fireEvent.click(await screen.findByLabelText('Inspect tool p_data_fetch'));
    fireEvent.change(await screen.findByLabelText('Argument defaults for p_data_fetch'), { target: { value: '{"query":"edited query"}' } });
    fireEvent.click(screen.getByText('Apply defaults'));
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    const expected = { ...withControls, stages: [withControls.stages[0], {
      ...withControls.stages[1], tool_defaults: { p_data_fetch: { query: 'edited query' } },
    }] };
    await waitFor(() => expect(apiClient.updateSavedPipeline).toHaveBeenCalledWith('p-1', { spec: expected }));
    await waitFor(() => expect(apiClient.listSavedPipelines).toHaveBeenCalledTimes(2));
    typeSpec('{"stages":[]}');
    fireEvent.click(screen.getByText('Attention survey'));
    expect(JSON.parse((screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement).value)).toEqual(expected);
  });

  it('clears a saved run binding preview when the tool catalog refreshes', async () => {
    apiClient.listSavedPipelines.mockResolvedValue([{ ...stored, spec: withControls }]);
    apiClient.previewPipelineBindings.mockResolvedValue({ root_job_id: 'job-7', stage: 'consume', omitted: 0, items: [{
      tool: 'p_data_fetch', argument: 'query', source_stage: 'produce', path: '/query',
      status: 'resolved', value_type: 'string', value_json: '"previous catalog value"',
    }] });
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));
    fireEvent.click(document.querySelector('.react-flow__node[data-id="consume"]')!);
    await waitFor(() => expect(screen.getByText('Preview applied bindings')).toBeEnabled());
    fireEvent.click(screen.getByText('Preview applied bindings'));
    await screen.findByText('"previous catalog value"');
    fireEvent.click(screen.getByRole('button', { name: 'Refresh tools and skills' }));
    expect(screen.queryByText('"previous catalog value"')).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Preview applied bindings')).toBeEnabled());
    expect(screen.queryByText('"previous catalog value"')).not.toBeInTheDocument();
    expect(apiClient.previewPipelineBindings).toHaveBeenCalledTimes(1);
  });

  it('lists what has been saved, with how often it has run', async () => {
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    render(<PipelineStudioPage />);

    expect(await screen.findByText('Attention survey')).toBeInTheDocument();
    expect(screen.getByText('×3')).toBeInTheDocument();
  });

  it('re-checks a pipeline it opens rather than trusting the stored verdict', async () => {
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));

    // The saved verdict was 'valid' when it was saved. Tools and their costs
    // move underneath a stored spec, so it is asked again.
    await waitFor(() =>
      expect(apiClient.checkPipeline).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'attention-survey' }),
        undefined
      )
    );
  });

  it('records which saved pipeline a launch came from', async () => {
    jest.spyOn(window, 'confirm').mockReturnValue(true);
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    apiClient.launchPipeline.mockResolvedValue({
      job_id: 'job-9',
      name: 'x',
      stages: [],
      estimated_seconds: 180,
      checkpoints: [],
    });
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));
    await screen.findAllByText('profile_c_workload');

    fireEvent.click(screen.getByRole('button', { name: /Launch/i }));

    await waitFor(() =>
      expect(apiClient.launchPipeline).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({ pipelineId: 'p-1' })
      )
    );
  });

  it('warns that deleting a pipeline keeps its runs', async () => {
    const confirm = jest.spyOn(window, 'confirm').mockReturnValue(false);
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    render(<PipelineStudioPage />);
    await screen.findByText('Attention survey');

    fireEvent.click(screen.getByRole('button', { name: /Delete Attention survey/i }));

    expect(confirm.mock.calls[0][0]).toContain('runs it started stay');
    expect(apiClient.deleteSavedPipeline).not.toHaveBeenCalled();
    confirm.mockRestore();
  });
});

describe('the starters', () => {
  beforeEach(() => {
    apiClient.listSavedPipelines.mockResolvedValue([]);
    apiClient.checkPipeline.mockResolvedValue(okCheck);
    window.localStorage.clear();
  });

  it('offers a reproduction pipeline as well as the default', () => {
    render(<PipelineStudioPage />);
    const picker = screen.getByLabelText('Start from a worked example');
    expect(picker).toHaveTextContent('Reproduce a paper');
  });

  it('loads a starter that is itself a valid pipeline', async () => {
    // The studio's copy of this spec is a duplicate of the one the backend
    // suite validates, so a typo here would ship a starter that fails its own
    // check — which teaches the wrong thing about the format on first contact.
    render(<PipelineStudioPage />);
    fireEvent.change(screen.getByLabelText('Start from a worked example'), {
      target: { value: 'Reproduce a paper' },
    });

    await waitFor(() => {
      const sent = apiClient.checkPipeline.mock.calls.at(-1)?.[0];
      expect(sent).toBeDefined();
      expect(sent.name).toBe('reproduce-paper-algorithm');
      expect(sent.stages.map((s: { id: string }) => s.id)).toEqual([
        'find',
        'specify',
        'implement',
        'measure',
        'compare',
      ]);
    });
  });

  it('asks for a verified implementation, not merely one that compiles', async () => {
    // The distinction the whole chain rests on: timing unchecked code produces
    // an accurate number for work nobody looked at.
    render(<PipelineStudioPage />);
    fireEvent.change(screen.getByLabelText('Start from a worked example'), {
      target: { value: 'Reproduce a paper' },
    });

    await waitFor(() => {
      const sent = apiClient.checkPipeline.mock.calls.at(-1)?.[0];
      const implement = sent.stages.find((s: { id: string }) => s.id === 'implement');
      expect(implement.contract.required_finding_types).toContain(
        'implementation_verified',
      );
      // And it loops: writing an algorithm from prose does not work first try.
      expect(implement.loop.max_iterations).toBeGreaterThan(1);
    });
  });
});


describe('importing a job chain', () => {
  const convertible = {
    chain_id: 'chain-1',
    name: 'literature_review_pipeline',
    steps: 4,
    convertible: true,
    blockers: [],
    contracts_to_write: 4,
    variables: [],
  };
  const blocked = {
    chain_id: 'chain-2',
    name: 'continuous_monitoring_with_alerts',
    steps: 2,
    convertible: false,
    blockers: [
      {
        step: 'Topic Monitoring',
        trigger: 'on_findings',
        reason: 'fires while the parent is still running',
      },
    ],
    contracts_to_write: 0,
    variables: [],
  };

  it('says nothing at all when there are no chains to import', async () => {
    render(<PipelineStudioPage />);
    await waitFor(() => expect(apiClient.surveyChainsForImport).toHaveBeenCalled());
    expect(screen.queryByLabelText('Job chains you can import')).not.toBeInTheDocument();
  });

  it('says how much work a convertible chain leaves behind', async () => {
    apiClient.surveyChainsForImport.mockResolvedValue({ candidates: [convertible] });
    render(<PipelineStudioPage />);

    expect(await screen.findByText('literature_review_pipeline')).toBeInTheDocument();
    // The number is the point: importing is faithful, not finished.
    expect(screen.getByText('4 contracts to write')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Import' })).toBeInTheDocument();
  });

  it('names the blocking step instead of offering to import', async () => {
    apiClient.surveyChainsForImport.mockResolvedValue({ candidates: [blocked] });
    render(<PipelineStudioPage />);

    expect(await screen.findByText(/Topic Monitoring needs on_findings/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Import' })).not.toBeInTheDocument();
  });

  it('survives a server that does not send variables at all', async () => {
    // The field was added after this endpoint shipped, and a running server can
    // be older than the page. TypeScript's `variables: string[]` is erased at
    // runtime, so the absent case has to be tested, not declared. This crashed
    // the whole studio with "undefined is not an object" until it was guarded.
    const { variables, ...withoutVariables } = { ...convertible };
    apiClient.surveyChainsForImport.mockResolvedValue({
      candidates: [withoutVariables],
    });
    render(<PipelineStudioPage />);

    expect(await screen.findByText('literature_review_pipeline')).toBeInTheDocument();
    // Nothing to ask for, so it imports directly.
    expect(screen.getByRole('button', { name: 'Import' })).toBeEnabled();
  });

  it('will not import a templated chain until its variables are given', async () => {
    // A chain fills {topic} at launch; a pipeline goal is literal. Importing
    // without a value would leave the braces in the goal.
    apiClient.surveyChainsForImport.mockResolvedValue({
      candidates: [{ ...convertible, variables: ['topic'] }],
    });
    render(<PipelineStudioPage />);

    const importButton = await screen.findByRole('button', { name: 'Import' });
    expect(importButton).toBeDisabled();

    fireEvent.change(
      screen.getByLabelText('topic for literature_review_pipeline'),
      { target: { value: 'attention sparsity' } }
    );
    expect(screen.getByRole('button', { name: 'Import' })).toBeEnabled();
  });

  it('sends the variables it collected', async () => {
    apiClient.surveyChainsForImport.mockResolvedValue({
      candidates: [{ ...convertible, variables: ['topic'] }],
    });
    apiClient.importChainAsPipeline.mockResolvedValue({
      id: 'p1', name: 'lit', spec: { name: 'lit', stages: [] }, last_check_valid: 'invalid',
    });
    render(<PipelineStudioPage />);

    fireEvent.change(
      await screen.findByLabelText('topic for literature_review_pipeline'),
      { target: { value: 'attention sparsity' } }
    );
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));

    await waitFor(() =>
      expect(apiClient.importChainAsPipeline).toHaveBeenCalledWith(
        'chain-1',
        undefined,
        { topic: 'attention sparsity' }
      )
    );
  });

  it('opens the imported pipeline in the editor', async () => {
    apiClient.surveyChainsForImport.mockResolvedValue({ candidates: [convertible] });
    apiClient.importChainAsPipeline.mockResolvedValue({
      id: 'pipeline-9',
      name: 'literature_review_pipeline',
      spec: { name: 'literature_review_pipeline', stages: [{ id: 'find', goal: 'Find' }] },
      last_check_valid: 'invalid',
    });
    apiClient.listSavedPipelines.mockResolvedValue([]);
    render(<PipelineStudioPage />);

    fireEvent.click(await screen.findByRole('button', { name: 'Import' }));

    await waitFor(() =>
      expect(apiClient.importChainAsPipeline).toHaveBeenCalledWith(
        'chain-1',
        undefined,
        undefined
      )
    );
    // Straight into the editor, because the contracts are what happens next.
    await waitFor(() => {
      const editor = screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement;
      expect(editor.value).toContain('literature_review_pipeline');
    });
  });
});

describe('suggesting what evidence a stage needs', () => {
  const SPEC = JSON.stringify({
    name: 'lit',
    stages: [{ id: 'gaps', goal: 'Identify research gaps', job_type: 'analysis' }],
  });

  it('offers the evidence with the words that matched, so a weak one is visible', async () => {
    apiClient.suggestContracts.mockResolvedValue({
      suggestions: {
        gaps: [
          { finding_type: 'research_gap', produced_by: 'identify_research_gaps',
            matched: ['research', 'gap'], typical_seconds: 60, perishable: false },
          { finding_type: 'research_graph', produced_by: 'build_research_graph',
            matched: ['research'], typical_seconds: 90, perishable: false },
        ],
      },
    });
    render(<PipelineStudioPage />);
    typeSpec(SPEC);

    expect(await screen.findByText('research_gap')).toBeInTheDocument();
    // The one-word match is shown as such rather than looking equally good.
    expect(screen.getByTitle(/matched research$/)).toBeInTheDocument();
  });

  it('writes the chosen evidence into that stage and leaves the rest alone', async () => {
    apiClient.suggestContracts.mockResolvedValue({
      suggestions: {
        gaps: [
          { finding_type: 'research_gap', produced_by: 'identify_research_gaps',
            matched: ['gap'], typical_seconds: 60, perishable: false },
        ],
      },
    });
    render(<PipelineStudioPage />);
    typeSpec(SPEC);

    fireEvent.click(await screen.findByText('research_gap'));

    const editor = screen.getByLabelText('Pipeline specification') as HTMLTextAreaElement;
    const spec = JSON.parse(editor.value);
    expect(spec.stages[0].contract.required_finding_types).toEqual(['research_gap']);
    // Nothing else about the stage was rewritten.
    expect(spec.stages[0].goal).toBe('Identify research gaps');
    expect(spec.name).toBe('lit');
  });

  it('says nothing when every stage already has a contract', async () => {
    apiClient.suggestContracts.mockResolvedValue({ suggestions: {} });
    render(<PipelineStudioPage />);
    typeSpec(SPEC);

    await waitFor(() => expect(apiClient.suggestContracts).toHaveBeenCalled());
    expect(screen.queryByText('Evidence these stages might need')).not.toBeInTheDocument();
  });
});
