/**
 * The folder tree, and the distinctions it has to hold onto.
 *
 * Two halves that look alike: computed system folders that cannot be edited or
 * filed into, and user folders that can. Most of what can go wrong here is one
 * half behaving like the other.
 */

import { fireEvent, render, screen, within } from '@testing-library/react';
import React from 'react';

import FolderTree from '../FolderTree';
import type { DocumentFolderNode } from '../../../types';

const node = (over: Partial<DocumentFolderNode>): DocumentFolderNode => ({
  key: 'k',
  name: 'n',
  kind: 'user',
  document_count: 0,
  subtree_count: 0,
  children: [],
  ...over,
});

const SYSTEM: DocumentFolderNode[] = [
  node({ key: 'all', name: 'All documents', kind: 'system', document_count: 29, subtree_count: 29 }),
  node({
    key: 'group:source',
    name: 'By source',
    kind: 'group',
    subtree_count: 16,
    children: [
      node({ key: 'source:s1', name: 'File Upload', kind: 'system', document_count: 8, subtree_count: 8 }),
      node({ key: 'source:s2', name: 'ArXiv', kind: 'system', document_count: 8, subtree_count: 8 }),
    ],
  }),
  node({ key: 'unfiled', name: 'Unfiled', kind: 'system', document_count: 29, subtree_count: 29 }),
];

const FOLDERS: DocumentFolderNode[] = [
  node({
    key: 'user:f1',
    id: 'f1',
    name: 'Microarchitecture',
    document_count: 1,
    subtree_count: 3,
    children: [node({ key: 'user:f2', id: 'f2', name: 'INT8', document_count: 2, subtree_count: 2 })],
  }),
];

const renderTree = (over: Partial<React.ComponentProps<typeof FolderTree>> = {}) => {
  const props = {
    system: SYSTEM,
    folders: FOLDERS,
    selectedKey: 'all',
    onSelect: jest.fn(),
    onCreate: jest.fn(),
    onRename: jest.fn(),
    onDelete: jest.fn(),
    onDropDocuments: jest.fn(),
    ...over,
  };
  return { ...render(<FolderTree {...props} />), props };
};

beforeEach(() => {
  window.localStorage.clear();
});

it('selects by key, which is what the documents list takes', () => {
  const { props } = renderTree();

  fireEvent.click(screen.getByText('Unfiled'));

  // Not an id: system folders have none, so the key is the only address that
  // works for both halves of the tree.
  expect(props.onSelect).toHaveBeenCalledWith('unfiled', expect.objectContaining({ key: 'unfiled' }));
});

it('does not filter to a group heading, it toggles it', () => {
  const { props } = renderTree();

  // 'By source' holds folders but contains no documents of its own, so
  // selecting it would show an empty list. Clicking it opens or closes it
  // instead. It starts open, so this click closes it.
  expect(screen.getByText('File Upload')).toBeInTheDocument();

  fireEvent.click(screen.getByText('By source'));

  expect(props.onSelect).not.toHaveBeenCalled();
  expect(screen.queryByText('File Upload')).not.toBeInTheDocument();

  fireEvent.click(screen.getByText('By source'));
  expect(screen.getByText('File Upload')).toBeInTheDocument();
  expect(props.onSelect).not.toHaveBeenCalled();
});

it('shows the subtree total on a folder that has children', () => {
  renderTree();

  // Microarchitecture holds 1 itself and 3 including its child: a number
  // beside a closed folder should mean everything inside it.
  const row = screen.getByRole('treeitem', { name: 'Microarchitecture' });
  expect(within(row).getByText('3')).toBeInTheDocument();
});

it('marks the selected folder for assistive technology too', () => {
  renderTree({ selectedKey: 'user:f1' });

  expect(
    screen.getByRole('treeitem', { name: 'Microarchitecture', selected: true })
  ).toBeInTheDocument();
  // A group heading is not selectable, so it carries no selected state at all.
  const group = screen.getByRole('treeitem', { name: 'By source' });
  expect(group).not.toHaveAttribute('aria-selected');
  expect(group).toHaveAttribute('aria-expanded', 'true');
});

it('offers actions on user folders and none on computed ones', () => {
  renderTree();

  expect(
    screen.getByRole('button', { name: 'Actions for Microarchitecture' })
  ).toBeInTheDocument();
  // A computed folder's membership is a query; there is nothing to rename or
  // file into.
  expect(screen.queryByRole('button', { name: 'Actions for Unfiled' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Actions for ArXiv' })).not.toBeInTheDocument();
});

it('accepts a document dropped on a user folder', () => {
  const { props } = renderTree();
  const row = screen.getByRole('treeitem', { name: 'Microarchitecture' });

  fireEvent.drop(row, {
    dataTransfer: {
      getData: () => JSON.stringify(['doc-1', 'doc-2']),
    },
  });

  expect(props.onDropDocuments).toHaveBeenCalledWith('f1', ['doc-1', 'doc-2']);
});

it('ignores a drop carrying something that is not document ids', () => {
  const { props } = renderTree();
  const row = screen.getByRole('treeitem', { name: 'Microarchitecture' });

  // A drag from elsewhere in the page: it must not throw, and must not file
  // anything.
  fireEvent.drop(row, { dataTransfer: { getData: () => 'not json' } });

  expect(props.onDropDocuments).not.toHaveBeenCalled();
});

it('remembers which folders were open', () => {
  const { unmount } = renderTree();

  // 'By source' opens by default; collapse it.
  fireEvent.click(screen.getByRole('button', { name: 'Collapse By source' }));
  expect(screen.queryByText('File Upload')).not.toBeInTheDocument();
  unmount();

  renderTree();
  expect(screen.queryByText('File Upload')).not.toBeInTheDocument();
});

it('says so when there are no user folders, without implying nothing is there', () => {
  renderTree({ folders: [] });

  expect(
    screen.getByText('No folders yet. Everything is still reachable above.')
  ).toBeInTheDocument();
});

it('announces loading rather than showing an empty tree', () => {
  renderTree({ loading: true });

  const status = screen.getByRole('status');
  expect(status).toHaveAttribute('aria-busy', 'true');
  expect(screen.getByText('Loading folders')).toBeInTheDocument();
  // Not an empty tree, which would read as "you have nothing".
  expect(screen.queryByRole('treeitem')).not.toBeInTheDocument();
});
