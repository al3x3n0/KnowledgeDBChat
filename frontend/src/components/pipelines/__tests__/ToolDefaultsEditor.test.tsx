import { fireEvent, render, screen } from '@testing-library/react';
import React, { useState } from 'react';

import ToolDefaultsEditor from '../ToolDefaultsEditor';

function Editor() {
  const [defaults, setDefaults] = useState<Record<string, unknown> | undefined>({ limit: 5 });
  return <>
    <ToolDefaultsEditor name="fetch" defaults={defaults} onChange={setDefaults} />
    <output data-testid="saved">{JSON.stringify(defaults ?? null)}</output>
  </>;
}

it('applies structured defaults only after confirmation and can clear them', () => {
  render(<Editor />);
  fireEvent.change(screen.getByLabelText('Argument defaults for fetch'), { target: { value: '{"limit":0,"options":{"fast":false}}' } });
  expect(screen.getByTestId('saved')).toHaveTextContent('{"limit":5}');
  expect(screen.getByText('Changes not applied.')).toBeInTheDocument();
  fireEvent.click(screen.getByText('Apply defaults'));
  expect(screen.getByTestId('saved')).toHaveTextContent('{"limit":0,"options":{"fast":false}}');
  fireEvent.click(screen.getByText('Clear defaults'));
  expect(screen.getByTestId('saved')).toHaveTextContent('null');
});

it.each(['{bad json', 'null', '[]', 'false'])('rejects invalid argument objects: %s', (value) => {
  render(<Editor />);
  fireEvent.change(screen.getByLabelText('Argument defaults for fetch'), { target: { value } });
  fireEvent.click(screen.getByText('Apply defaults'));
  expect(screen.getByRole('alert')).toBeInTheDocument();
  expect(screen.getByTestId('saved')).toHaveTextContent('{"limit":5}');
});
