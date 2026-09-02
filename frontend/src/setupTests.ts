// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';

// testing-library gives every findBy* and waitFor one second by default. The
// heavier pages here mount, fire several queries and settle well inside that
// on an idle machine and not reliably on a loaded one, which is what made
// AutonomousAgentsPage's shards fail intermittently: 'auto-applies the default
// trace view' failed looking for an option that arrived at 1.1 seconds.
//
// This weakens no assertion. A test that passes in 200ms still passes in
// 200ms; only a failing one waits longer before saying so.
configure({ asyncUtilTimeout: 5000 });

// CRA/Jest (react-scripts) doesn't transform ESM in node_modules by default.
// Some dependencies we use (react-markdown, tiptap v3) ship ESM and will fail
// to parse in tests unless mocked.
jest.mock('react-markdown', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: ({ children }: any) =>
      React.createElement('div', { 'data-testid': 'react-markdown' }, children),
  };
});

jest.mock('remark-gfm', () => ({
  __esModule: true,
  default: () => null,
}));

jest.mock('@tiptap/react', () => ({
  __esModule: true,
  useEditor: () => null,
  EditorContent: () => null,
}));

jest.mock('@tiptap/starter-kit', () => ({
  __esModule: true,
  default: {},
}));

jest.mock('@tiptap/extension-underline', () => ({
  __esModule: true,
  Underline: {},
}));

jest.mock('@tiptap/extension-table', () => ({
  __esModule: true,
  Table: {},
}));

jest.mock('@tiptap/extension-table-row', () => ({
  __esModule: true,
  TableRow: {},
}));

jest.mock('@tiptap/extension-table-header', () => ({
  __esModule: true,
  TableHeader: {},
}));

jest.mock('@tiptap/extension-table-cell', () => ({
  __esModule: true,
  TableCell: {},
}));

jest.mock('axios', () => {
  const mockInstance = {
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  };
  return {
    __esModule: true,
    default: {
      create: jest.fn(() => mockInstance),
    },
  };
});
