/**
 * Full-screen modal for editing DOCX documents
 */

import React, { useState, useEffect, useCallback } from 'react';
import { X, Save, Loader2, AlertTriangle } from 'lucide-react';
import toast from 'react-hot-toast';
import DocxEditor from './DocxEditor';
import { apiClient } from '../../services/api';
import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';

interface DocxEditorModalProps {
  documentId: string;
  documentTitle: string;
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

const DocxEditorModal: React.FC<DocxEditorModalProps> = ({
  documentId,
  documentTitle,
  isOpen,
  onClose,
  onSaved,
}) => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [htmlContent, setHtmlContent] = useState<string>('');
  const [originalContent, setOriginalContent] = useState<string>('');
  const [version, setVersion] = useState<string>('');
  const [warnings, setWarnings] = useState<string[]>([]);
  const [hasChanges, setHasChanges] = useState(false);

  // Load document content
  useEffect(() => {
    if (isOpen && documentId) {
      loadDocument();
    }
  }, [isOpen, documentId]);

  const loadDocument = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.getDocumentForEditing(documentId);
      setHtmlContent(response.html_content);
      setOriginalContent(response.html_content);
      setVersion(response.version);
      setWarnings(response.warnings || []);
      setHasChanges(false);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to load document';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleContentChange = useCallback((newHtml: string) => {
    setHtmlContent(newHtml);
    setHasChanges(newHtml !== originalContent);
  }, [originalContent]);

  const handleSave = async () => {
    if (!hasChanges) {
      toast.success('No changes to save');
      return;
    }

    setSaving(true);
    try {
      const response = await apiClient.saveDocumentEdits(documentId, {
        html_content: htmlContent,
        version: version,
        create_backup: true,
      });

      if (response.success) {
        toast.success('Document saved successfully');
        setVersion(response.new_version);
        setOriginalContent(htmlContent);
        setHasChanges(false);
        if (onSaved) {
          onSaved();
        }
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to save document';
      toast.error(errorMsg);
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    if (hasChanges) {
      if (!window.confirm('You have unsaved changes. Are you sure you want to close?')) {
        return;
      }
    }
    onClose();
  };

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      // Ctrl/Cmd + S to save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }

      // Escape to close
      if (e.key === 'Escape') {
        handleClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, hasChanges, htmlContent, version]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-gray-900 truncate max-w-md">
            {documentTitle}
          </h2>
          {hasChanges && (
            <span className="text-xs text-orange-600 bg-orange-100 px-2 py-0.5 rounded">
              Unsaved changes
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={handleSave}
            disabled={saving || !hasChanges}
            className="flex items-center gap-2"
          >
            {saving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Save
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClose}
            className="flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            Close
          </Button>
        </div>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="px-4 py-2 bg-yellow-50 border-b border-yellow-200">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-yellow-800">
              <p className="font-medium">Conversion warnings:</p>
              <ul className="list-disc list-inside mt-1">
                {warnings.slice(0, 3).map((warning, idx) => (
                  <li key={idx}>{warning}</li>
                ))}
                {warnings.length > 3 && (
                  <li>... and {warnings.length - 3} more</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Editor Content */}
      <div className="flex-1 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <LoadingSpinner size="lg" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full text-red-600">
            <AlertTriangle className="w-12 h-12 mb-4" />
            <p className="text-lg font-medium">Failed to load document</p>
            <p className="text-sm text-gray-500 mt-1">{error}</p>
            <Button variant="secondary" onClick={loadDocument} className="mt-4">
              Try Again
            </Button>
          </div>
        ) : (
          <DocxEditor
            initialContent={htmlContent}
            onChange={handleContentChange}
            readOnly={false}
          />
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-200 bg-gray-50 text-xs text-gray-500 flex items-center justify-between">
        <span>Press Ctrl+S to save, Escape to close</span>
        <span>Version: {version || 'N/A'}</span>
      </div>
    </div>
  );
};

export default DocxEditorModal;
