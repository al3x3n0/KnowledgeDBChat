/**
 * TipTap-based DOCX editor component
 */

import React from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import { Underline } from '@tiptap/extension-underline';
import { Table } from '@tiptap/extension-table';
import { TableRow } from '@tiptap/extension-table-row';
import { TableCell } from '@tiptap/extension-table-cell';
import { TableHeader } from '@tiptap/extension-table-header';
import EditorToolbar from './EditorToolbar';

interface DocxEditorProps {
  initialContent: string;
  onChange?: (html: string) => void;
  readOnly?: boolean;
}

const DocxEditor: React.FC<DocxEditorProps> = ({
  initialContent,
  onChange,
  readOnly = false,
}) => {
  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3, 4, 5, 6],
        },
      }),
      Underline,
      Table.configure({
        resizable: true,
      }),
      TableRow,
      TableHeader,
      TableCell,
    ],
    content: initialContent,
    editable: !readOnly,
    onUpdate: ({ editor }) => {
      if (onChange) {
        onChange(editor.getHTML());
      }
    },
  });

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-sm border border-gray-200">
      {!readOnly && <EditorToolbar editor={editor} />}
      <div className="flex-1 overflow-auto p-4">
        <EditorContent
          editor={editor}
          className="prose prose-sm max-w-none min-h-full focus:outline-none"
        />
      </div>
      <style>{`
        .ProseMirror {
          min-height: 100%;
          padding: 1rem;
          outline: none;
        }
        .ProseMirror p {
          margin: 0.5em 0;
        }
        .ProseMirror h1 {
          font-size: 2em;
          font-weight: bold;
          margin: 0.67em 0;
        }
        .ProseMirror h2 {
          font-size: 1.5em;
          font-weight: bold;
          margin: 0.75em 0;
        }
        .ProseMirror h3 {
          font-size: 1.17em;
          font-weight: bold;
          margin: 0.83em 0;
        }
        .ProseMirror ul, .ProseMirror ol {
          padding-left: 1.5em;
          margin: 0.5em 0;
        }
        .ProseMirror li {
          margin: 0.25em 0;
        }
        .ProseMirror table {
          border-collapse: collapse;
          width: 100%;
          margin: 1em 0;
        }
        .ProseMirror th, .ProseMirror td {
          border: 1px solid #ccc;
          padding: 0.5em;
          text-align: left;
        }
        .ProseMirror th {
          background-color: #f5f5f5;
          font-weight: bold;
        }
        .ProseMirror blockquote {
          border-left: 3px solid #ccc;
          margin: 1em 0;
          padding-left: 1em;
          color: #666;
        }
        .ProseMirror-selectednode {
          outline: 2px solid #68cef8;
        }
      `}</style>
    </div>
  );
};

export default DocxEditor;
