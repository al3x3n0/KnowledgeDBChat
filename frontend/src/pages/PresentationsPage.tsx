/**
 * AI PowerPoint Presentation Generator Page
 */

import React, { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import apiClient from '../services/api';
import { PresentationJob, PresentationStatus, PresentationStyle, Document, PresentationTemplate, ThemeConfig } from '../types';
import { SlidePreviewModal } from '../components/presentations';

// Status badge colors
const STATUS_COLORS: Record<PresentationStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  generating: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-gray-100 text-gray-800',
};

// Stage labels for display
const STAGE_LABELS: Record<string, string> = {
  starting: 'Starting...',
  gathering_context: 'Gathering context from documents...',
  context_gathered: 'Context gathered',
  generating_outline: 'Generating presentation outline...',
  outline_generated: 'Outline generated',
  generating_content: 'Generating slide content...',
  generating_slides: 'Generating slides...',
  content_generated: 'Content generated',
  rendering_diagrams: 'Rendering diagrams...',
  diagrams_rendered: 'Diagrams rendered',
  skipped_diagrams: 'Skipped diagrams',
  building_pptx: 'Building PowerPoint file...',
  pptx_built: 'PowerPoint built',
  uploading: 'Uploading file...',
  completed: 'Completed',
};

interface CreateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (data: CreatePresentationData) => void;
  documents: Document[];
  templates: PresentationTemplate[];
  isLoading: boolean;
}

interface CreatePresentationData {
  title: string;
  topic: string;
  source_document_ids: string[];
  slide_count: number;
  style: PresentationStyle;
  include_diagrams: boolean;
  template_id?: string;
  custom_theme?: ThemeConfig;
}

// All available styles
const AVAILABLE_STYLES: { value: PresentationStyle; label: string; description: string }[] = [
  { value: 'professional', label: 'Professional', description: 'Clean, corporate look with dark blue accents' },
  { value: 'casual', label: 'Casual', description: 'Friendly and approachable with warm colors' },
  { value: 'technical', label: 'Technical', description: 'Developer-focused with monospace fonts' },
  { value: 'modern', label: 'Modern', description: 'Contemporary design with bold contrasts' },
  { value: 'minimal', label: 'Minimal', description: 'Simple and clean with subtle grays' },
  { value: 'corporate', label: 'Corporate', description: 'Traditional business style with orange accents' },
  { value: 'creative', label: 'Creative', description: 'Artistic with purple and turquoise' },
  { value: 'dark', label: 'Dark', description: 'Dark theme for low-light presentations' },
];

// Template Upload Modal
interface TemplateUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (template: PresentationTemplate) => void;
}

const TemplateUploadModal: React.FC<TemplateUploadModalProps> = ({
  isOpen,
  onClose,
  onUpload,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isPublic, setIsPublic] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith('.pptx')) {
        setFile(droppedFile);
        if (!name) {
          setName(droppedFile.name.replace(/\.pptx$/i, ''));
        }
      } else {
        toast.error('Only .pptx files are allowed');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.toLowerCase().endsWith('.pptx')) {
        setFile(selectedFile);
        if (!name) {
          setName(selectedFile.name.replace(/\.pptx$/i, ''));
        }
      } else {
        toast.error('Only .pptx files are allowed');
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !name.trim()) {
      toast.error('Please select a file and provide a name');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const template = await apiClient.uploadPresentationTemplate(
        file,
        name.trim(),
        description.trim() || undefined,
        isPublic,
        (progress) => setUploadProgress(progress)
      );
      toast.success('Template uploaded successfully!');
      onUpload(template);
      // Reset form
      setFile(null);
      setName('');
      setDescription('');
      setIsPublic(false);
      onClose();
    } catch (error: any) {
      console.error('Upload failed:', error);
      toast.error(error.response?.data?.detail || 'Failed to upload template');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900">Upload PPTX Template</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            disabled={isUploading}
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6">
          {/* File Drop Zone */}
          <div
            className={`mb-4 border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
              dragActive
                ? 'border-blue-500 bg-blue-50'
                : file
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="flex items-center justify-center space-x-3">
                <svg className="w-8 h-8 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="text-left">
                  <p className="text-sm font-medium text-gray-900">{file.name}</p>
                  <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="text-red-500 hover:text-red-700"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ) : (
              <>
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-2 text-sm text-gray-600">
                  Drag and drop your .pptx file here, or{' '}
                  <label className="text-blue-600 hover:text-blue-500 cursor-pointer">
                    browse
                    <input
                      type="file"
                      className="hidden"
                      accept=".pptx"
                      onChange={handleFileChange}
                      disabled={isUploading}
                    />
                  </label>
                </p>
                <p className="mt-1 text-xs text-gray-500">PPTX files up to 50MB</p>
              </>
            )}
          </div>

          {/* Upload Progress */}
          {isUploading && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-gray-600">Uploading...</span>
                <span className="text-sm text-gray-600">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Template Name */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">Template Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Custom Template"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isUploading}
              required
            />
          </div>

          {/* Description */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description of this template..."
              rows={2}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isUploading}
            />
          </div>

          {/* Public Toggle */}
          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                disabled={isUploading}
              />
              <span className="ml-2 text-sm text-gray-700">Make template public (visible to all users)</span>
            </label>
          </div>

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              disabled={isUploading}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isUploading || !file || !name.trim()}
              className="px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              {isUploading && (
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              Upload Template
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

const CreatePresentationModal: React.FC<CreateModalProps> = ({
  isOpen,
  onClose,
  onCreate,
  documents,
  templates,
  isLoading,
}) => {
  const [title, setTitle] = useState('');
  const [topic, setTopic] = useState('');
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [slideCount, setSlideCount] = useState(10);
  const [style, setStyle] = useState<PresentationStyle>('professional');
  const [includeDiagrams, setIncludeDiagrams] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [useCustomTheme, setUseCustomTheme] = useState(false);
  const [customColors, setCustomColors] = useState({
    title_color: '#1a365d',
    accent_color: '#2e86ab',
    text_color: '#333333',
    bg_color: '#ffffff',
  });

  // Get selected template for preview
  const selectedTemplate = templates.find(t => t.id === selectedTemplateId);

  const filteredDocs = documents.filter(
    (doc) =>
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.source_identifier?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !topic.trim()) {
      toast.error('Please fill in title and topic');
      return;
    }
    const data: CreatePresentationData = {
      title: title.trim(),
      topic: topic.trim(),
      source_document_ids: selectedDocs,
      slide_count: slideCount,
      style,
      include_diagrams: includeDiagrams,
    };

    // Add template or custom theme
    if (selectedTemplateId) {
      data.template_id = selectedTemplateId;
    } else if (useCustomTheme) {
      data.custom_theme = {
        colors: customColors,
        fonts: { title_font: 'Calibri', body_font: 'Calibri' },
        sizes: { title_size: 44, subtitle_size: 24, heading_size: 36, body_size: 20, bullet_size: 18 },
      };
    }

    onCreate(data);
  };

  const toggleDoc = (docId: string) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900">Create AI Presentation</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            disabled={isLoading}
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
          {/* Title */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Presentation Title *
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Quarterly Review 2024"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            />
          </div>

          {/* Topic */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Topic / Description *
            </label>
            <textarea
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Describe what the presentation should cover..."
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            />
          </div>

          {/* Template Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Theme / Template</label>

            {/* Template Options */}
            <div className="space-y-2 mb-3">
              {/* Use Style Option */}
              <label className="flex items-center p-2 border rounded-md cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  checked={!selectedTemplateId && !useCustomTheme}
                  onChange={() => { setSelectedTemplateId(null); setUseCustomTheme(false); }}
                  className="text-blue-600 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <span className="ml-2 text-sm text-gray-700">Use built-in style</span>
              </label>

              {/* System Templates */}
              {templates.filter(t => t.is_system).length > 0 && (
                <div className="border rounded-md overflow-hidden">
                  <div className="px-3 py-2 bg-gray-50 text-xs font-medium text-gray-500 uppercase">
                    System Templates
                  </div>
                  <div className="max-h-32 overflow-y-auto">
                    {templates.filter(t => t.is_system).map(template => (
                      <label
                        key={template.id}
                        className={`flex items-center px-3 py-2 cursor-pointer hover:bg-gray-50 border-t ${
                          selectedTemplateId === template.id ? 'bg-blue-50' : ''
                        }`}
                      >
                        <input
                          type="radio"
                          checked={selectedTemplateId === template.id}
                          onChange={() => { setSelectedTemplateId(template.id); setUseCustomTheme(false); }}
                          className="text-blue-600 focus:ring-blue-500"
                          disabled={isLoading}
                        />
                        <div className="ml-2 flex-1">
                          <span className="text-sm font-medium text-gray-900">{template.name}</span>
                          {template.description && (
                            <p className="text-xs text-gray-500">{template.description}</p>
                          )}
                        </div>
                        {template.theme_config?.colors && (
                          <div className="flex space-x-1">
                            <div
                              className="w-4 h-4 rounded-full border"
                              style={{ backgroundColor: template.theme_config.colors.title_color }}
                              title="Title color"
                            />
                            <div
                              className="w-4 h-4 rounded-full border"
                              style={{ backgroundColor: template.theme_config.colors.accent_color }}
                              title="Accent color"
                            />
                          </div>
                        )}
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* User PPTX Templates */}
              {templates.filter(t => !t.is_system && t.template_type === 'pptx').length > 0 && (
                <div className="border rounded-md overflow-hidden">
                  <div className="px-3 py-2 bg-purple-50 text-xs font-medium text-purple-600 uppercase">
                    Your PPTX Templates
                  </div>
                  <div className="max-h-32 overflow-y-auto">
                    {templates.filter(t => !t.is_system && t.template_type === 'pptx').map(template => (
                      <label
                        key={template.id}
                        className={`flex items-center px-3 py-2 cursor-pointer hover:bg-gray-50 border-t ${
                          selectedTemplateId === template.id ? 'bg-blue-50' : ''
                        }`}
                      >
                        <input
                          type="radio"
                          checked={selectedTemplateId === template.id}
                          onChange={() => { setSelectedTemplateId(template.id); setUseCustomTheme(false); }}
                          className="text-blue-600 focus:ring-blue-500"
                          disabled={isLoading}
                        />
                        <div className="ml-2 flex-1">
                          <div className="flex items-center">
                            <span className="text-sm font-medium text-gray-900">{template.name}</span>
                            <span className="ml-2 px-1.5 py-0.5 text-xs bg-purple-100 text-purple-700 rounded">
                              PPTX
                            </span>
                          </div>
                          {template.description && (
                            <p className="text-xs text-gray-500">{template.description}</p>
                          )}
                        </div>
                        <svg className="w-5 h-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* Custom Theme Option */}
              <label className="flex items-center p-2 border rounded-md cursor-pointer hover:bg-gray-50">
                <input
                  type="radio"
                  checked={useCustomTheme}
                  onChange={() => { setSelectedTemplateId(null); setUseCustomTheme(true); }}
                  className="text-blue-600 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <span className="ml-2 text-sm text-gray-700">Custom colors</span>
              </label>
            </div>

            {/* Custom Color Pickers (shown when custom theme selected) */}
            {useCustomTheme && (
              <div className="p-3 bg-gray-50 rounded-md space-y-2">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Title Color</label>
                    <input
                      type="color"
                      value={customColors.title_color}
                      onChange={(e) => setCustomColors(prev => ({ ...prev, title_color: e.target.value }))}
                      className="w-full h-8 rounded cursor-pointer"
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Accent Color</label>
                    <input
                      type="color"
                      value={customColors.accent_color}
                      onChange={(e) => setCustomColors(prev => ({ ...prev, accent_color: e.target.value }))}
                      className="w-full h-8 rounded cursor-pointer"
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Text Color</label>
                    <input
                      type="color"
                      value={customColors.text_color}
                      onChange={(e) => setCustomColors(prev => ({ ...prev, text_color: e.target.value }))}
                      className="w-full h-8 rounded cursor-pointer"
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Background Color</label>
                    <input
                      type="color"
                      value={customColors.bg_color}
                      onChange={(e) => setCustomColors(prev => ({ ...prev, bg_color: e.target.value }))}
                      className="w-full h-8 rounded cursor-pointer"
                      disabled={isLoading}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Style (shown when using built-in style) */}
          {!selectedTemplateId && !useCustomTheme && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Style</label>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value as PresentationStyle)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                disabled={isLoading}
              >
                {AVAILABLE_STYLES.map(s => (
                  <option key={s.value} value={s.value}>{s.label} - {s.description}</option>
                ))}
              </select>
            </div>
          )}

          {/* Slide Count */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Slides: {slideCount}
            </label>
            <input
              type="range"
              min="5"
              max="20"
              value={slideCount}
              onChange={(e) => setSlideCount(parseInt(e.target.value))}
              className="w-full"
              disabled={isLoading}
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>5</span>
              <span>20</span>
            </div>
          </div>

          {/* Include Diagrams */}
          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeDiagrams}
                onChange={(e) => setIncludeDiagrams(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                disabled={isLoading}
              />
              <span className="ml-2 text-sm text-gray-700">Include diagrams (Mermaid)</span>
            </label>
          </div>

          {/* Source Documents */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Source Documents (optional - leave empty to search all)
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search documents..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md mb-2"
              disabled={isLoading}
            />
            <div className="border border-gray-200 rounded-md max-h-48 overflow-y-auto">
              {filteredDocs.length === 0 ? (
                <p className="p-3 text-sm text-gray-500">No documents found</p>
              ) : (
                filteredDocs.slice(0, 50).map((doc) => (
                  <label
                    key={doc.id}
                    className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                  >
                    <input
                      type="checkbox"
                      checked={selectedDocs.includes(doc.id)}
                      onChange={() => toggleDoc(doc.id)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      disabled={isLoading}
                    />
                    <div className="ml-3 overflow-hidden">
                      <p className="text-sm font-medium text-gray-900 truncate">{doc.title}</p>
                      <p className="text-xs text-gray-500 truncate">{doc.source_identifier}</p>
                    </div>
                  </label>
                ))
              )}
            </div>
            {selectedDocs.length > 0 && (
              <p className="mt-1 text-xs text-gray-500">{selectedDocs.length} document(s) selected</p>
            )}
          </div>
        </form>

        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-end space-x-3">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isLoading || !title.trim() || !topic.trim()}
            className="px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {isLoading && (
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            )}
            Generate Presentation
          </button>
        </div>
      </div>
    </div>
  );
};

const PresentationsPage: React.FC = () => {
  const [jobs, setJobs] = useState<PresentationJob[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [templates, setTemplates] = useState<PresentationTemplate[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [previewJob, setPreviewJob] = useState<PresentationJob | null>(null);
  const [activeWebSockets, setActiveWebSockets] = useState<Map<string, WebSocket>>(new Map());

  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    try {
      const data = await apiClient.listPresentations({ limit: 50 });
      setJobs(data);
    } catch (error) {
      console.error('Failed to fetch presentation jobs:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch documents for the modal
  const fetchDocuments = useCallback(async () => {
    try {
      const data = await apiClient.getDocuments({ limit: 500 });
      setDocuments(data);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  }, []);

  // Fetch templates for the modal
  const fetchTemplates = useCallback(async () => {
    try {
      const data = await apiClient.listPresentationTemplates();
      setTemplates(data);
    } catch (error) {
      console.error('Failed to fetch templates:', error);
    }
  }, []);

  useEffect(() => {
    fetchJobs();
    fetchDocuments();
    fetchTemplates();
  }, [fetchJobs, fetchDocuments, fetchTemplates]);

  // Setup WebSocket for in-progress jobs
  useEffect(() => {
    const inProgressJobs = jobs.filter((j) => j.status === 'pending' || j.status === 'generating');

    inProgressJobs.forEach((job) => {
      if (!activeWebSockets.has(job.id)) {
        try {
          const ws = apiClient.createPresentationProgressWebSocket(job.id);

          ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setJobs((prev) =>
              prev.map((j) =>
                j.id === job.id
                  ? {
                      ...j,
                      progress: data.progress,
                      current_stage: data.stage,
                      status: data.status,
                      error: data.error,
                    }
                  : j
              )
            );

            if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
              ws.close();
              setActiveWebSockets((prev) => {
                const newMap = new Map(prev);
                newMap.delete(job.id);
                return newMap;
              });
              // Refresh to get the full job data
              fetchJobs();
            }
          };

          ws.onerror = (error) => {
            console.error('WebSocket error:', error);
          };

          ws.onclose = () => {
            setActiveWebSockets((prev) => {
              const newMap = new Map(prev);
              newMap.delete(job.id);
              return newMap;
            });
          };

          setActiveWebSockets((prev) => new Map(prev).set(job.id, ws));
        } catch (error) {
          console.error('Failed to create WebSocket:', error);
        }
      }
    });

    // Cleanup old WebSockets
    return () => {
      activeWebSockets.forEach((ws) => ws.close());
    };
  }, [jobs, activeWebSockets, fetchJobs]);

  const handleCreate = async (data: CreatePresentationData) => {
    setIsCreating(true);
    try {
      const job = await apiClient.createPresentation({
        title: data.title,
        topic: data.topic,
        source_document_ids: data.source_document_ids,
        slide_count: data.slide_count,
        style: data.style,
        include_diagrams: data.include_diagrams,
        template_id: data.template_id,
        custom_theme: data.custom_theme,
      });
      setJobs((prev) => [job, ...prev]);
      setShowCreateModal(false);
      toast.success('Presentation generation started!');
    } catch (error) {
      console.error('Failed to create presentation:', error);
      toast.error('Failed to start presentation generation');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDownload = async (job: PresentationJob) => {
    try {
      await apiClient.downloadPresentation(job.id);
      toast.success('Download started');
    } catch (error) {
      console.error('Download failed:', error);
      toast.error('Download failed');
    }
  };

  const handleDelete = async (jobId: string) => {
    if (!window.confirm('Are you sure you want to delete this presentation?')) return;
    try {
      await apiClient.deletePresentation(jobId);
      setJobs((prev) => prev.filter((j) => j.id !== jobId));
      toast.success('Presentation deleted');
    } catch (error) {
      console.error('Delete failed:', error);
      toast.error('Delete failed');
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await apiClient.cancelPresentation(jobId);
      toast.success('Presentation generation cancelled');
      fetchJobs();
    } catch (error) {
      console.error('Cancel failed:', error);
      toast.error('Cancel failed');
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '-';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleTemplateUpload = (template: PresentationTemplate) => {
    setTemplates((prev) => [template, ...prev]);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">AI Presentations</h1>
            <p className="mt-1 text-sm text-gray-500">
              Generate PowerPoint presentations from your knowledge base using AI
            </p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={() => setShowUploadModal(true)}
              className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 flex items-center"
            >
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Upload Template
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
            >
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Create Presentation
            </button>
          </div>
        </div>

        {/* Jobs List */}
        {isLoading ? (
          <div className="flex justify-center py-12">
            <svg className="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">No presentations yet</h3>
            <p className="mt-1 text-sm text-gray-500">Get started by creating a new AI-generated presentation.</p>
            <div className="mt-6">
              <button
                onClick={() => setShowCreateModal(true)}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                Create your first presentation
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Title
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Style
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {jobs.map((job) => (
                  <tr key={job.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900 truncate max-w-xs">{job.title}</div>
                      <div className="text-xs text-gray-500 truncate max-w-xs">{job.topic}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          STATUS_COLORS[job.status]
                        }`}
                      >
                        {job.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {(job.status === 'pending' || job.status === 'generating') ? (
                        <div className="w-32">
                          <div className="flex items-center">
                            <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${job.progress}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-600">{job.progress}%</span>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {STAGE_LABELS[job.current_stage || ''] || job.current_stage || 'Waiting...'}
                          </div>
                        </div>
                      ) : job.status === 'failed' ? (
                        <span className="text-xs text-red-600" title={job.error}>
                          {job.error?.substring(0, 50)}...
                        </span>
                      ) : (
                        <span className="text-xs text-gray-500">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 capitalize">
                      {job.style}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatFileSize(job.file_size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(job.created_at)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium space-x-2">
                      {/* Preview button - available when outline is generated */}
                      {job.generated_outline && (
                        <button
                          onClick={() => setPreviewJob(job)}
                          className="text-purple-600 hover:text-purple-900"
                          title="Preview slides"
                        >
                          <svg className="w-5 h-5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                        </button>
                      )}
                      {job.status === 'completed' && (
                        <button
                          onClick={() => handleDownload(job)}
                          className="text-blue-600 hover:text-blue-900"
                          title="Download"
                        >
                          <svg className="w-5 h-5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                        </button>
                      )}
                      {(job.status === 'pending' || job.status === 'generating') && (
                        <button
                          onClick={() => handleCancel(job.id)}
                          className="text-yellow-600 hover:text-yellow-900"
                          title="Cancel"
                        >
                          <svg className="w-5 h-5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(job.id)}
                        className="text-red-600 hover:text-red-900"
                        title="Delete"
                      >
                        <svg className="w-5 h-5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Create Modal */}
        <CreatePresentationModal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          onCreate={handleCreate}
          documents={documents}
          templates={templates}
          isLoading={isCreating}
        />

        {/* Template Upload Modal */}
        <TemplateUploadModal
          isOpen={showUploadModal}
          onClose={() => setShowUploadModal(false)}
          onUpload={handleTemplateUpload}
        />

        {/* Slide Preview Modal */}
        {previewJob && (
          <SlidePreviewModal
            job={previewJob}
            isOpen={previewJob !== null}
            onClose={() => setPreviewJob(null)}
          />
        )}
    </div>
  );
};

export default PresentationsPage;
