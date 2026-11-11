import { fireEvent, render, screen } from '@testing-library/react';
import React, { useState } from 'react';
import ToolBindingsEditor, { ToolInputBinding } from '../ToolBindingsEditor';

function Editor() {
  const [bindings, setBindings] = useState<Record<string, ToolInputBinding> | undefined>();
  return <>
    <ToolBindingsEditor name="consume" arguments={['query']} sources={['produce']} bindings={bindings} onChange={setBindings} />
    <output data-testid="saved">{JSON.stringify(bindings ?? null)}</output>
  </>;
}

it('binds a named input to dependency results and removes it explicitly', () => {
  render(<Editor />);
  fireEvent.click(screen.getByText('Add binding'));
  expect(screen.getByTestId('saved')).toHaveTextContent('null');
  fireEvent.change(screen.getByLabelText('Binding path 1 for consume'), { target: { value: '/findings/0/value' } });
  fireEvent.click(screen.getByText('Apply bindings'));
  expect(screen.getByTestId('saved')).toHaveTextContent('{"query":{"stage":"produce","path":"/findings/0/value"}}');
  fireEvent.click(screen.getByLabelText('Remove binding 1 for consume'));
  fireEvent.click(screen.getByText('Apply bindings'));
  expect(screen.getByTestId('saved')).toHaveTextContent('null');
});

it('rejects expression syntax and malformed JSON Pointer escapes', () => {
  render(<Editor />);
  fireEvent.click(screen.getByText('Add binding'));
  for (const path of ['results.value', '/bad~escape']) {
    fireEvent.change(screen.getByLabelText('Binding path 1 for consume'), { target: { value: path } });
    fireEvent.click(screen.getByText('Apply bindings'));
    expect(screen.getByRole('alert')).toHaveTextContent('JSON Pointer');
    expect(screen.getByTestId('saved')).toHaveTextContent('null');
  }
});

it('does not offer bindings without a dependency stage', () => {
  render(<ToolBindingsEditor name="consume" arguments={['query']} sources={[]} onChange={jest.fn()} />);
  expect(screen.getByText('Add binding')).toBeDisabled();
  expect(screen.getByText(/Add a stage dependency/)).toBeInTheDocument();
});

it('flags a removed dependency immediately and retargets only when applied', () => {
  const saved = { query: { stage: 'produce', path: '/query' } };
  const onChange = jest.fn();
  const { rerender } = render(<ToolBindingsEditor name="consume" arguments={['query']} sources={['produce', 'other']} bindings={saved} onChange={onChange} />);
  rerender(<ToolBindingsEditor name="consume" arguments={['query']} sources={['other']} bindings={saved} onChange={onChange} />);
  expect(screen.getByRole('alert')).toHaveTextContent("Source 'produce' is not a dependency");
  expect(screen.getByLabelText('Binding source 1 for consume')).toHaveValue('produce');
  expect(screen.getByLabelText('Binding source 1 for consume')).toHaveAttribute('aria-invalid', 'true');
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText('Binding source 1 for consume'), { target: { value: 'other' } });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByText('Apply bindings'));
  expect(onChange).toHaveBeenCalledWith({ query: { stage: 'other', path: '/query' } });
  expect(saved.query.stage).toBe('produce');
});

it('keeps a binding after its last dependency is removed until removal is applied explicitly', () => {
  const onChange = jest.fn();
  render(<ToolBindingsEditor name="consume" arguments={['query']} sources={[]} bindings={{ query: { stage: 'produce', path: '' } }} onChange={onChange} />);
  expect(screen.getByRole('alert')).toHaveTextContent("Source 'produce' is not a dependency");
  expect(screen.getByText('Apply bindings')).toBeDisabled();
  fireEvent.click(screen.getByLabelText('Remove binding 1 for consume'));
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByText('Apply bindings'));
  expect(onChange).toHaveBeenCalledWith(undefined);
});

it('flags a changed tool schema and preserves the pointer while repairing the argument', () => {
  const onChange = jest.fn();
  render(<ToolBindingsEditor name="consume" arguments={['query_terms']} sources={['produce']} bindings={{ query: { stage: 'produce', path: '/findings/0' } }} onChange={onChange} />);
  expect(screen.getByRole('alert')).toHaveTextContent("Argument 'query' is no longer declared");
  fireEvent.change(screen.getByLabelText('Binding argument 1 for consume'), { target: { value: 'query_terms' } });
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByText('Apply bindings'));
  expect(onChange).toHaveBeenCalledWith({ query_terms: { stage: 'produce', path: '/findings/0' } });
});
