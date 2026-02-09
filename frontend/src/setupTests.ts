// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

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
