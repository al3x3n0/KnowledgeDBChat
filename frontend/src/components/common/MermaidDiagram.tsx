/**
 * Mermaid Diagram Renderer Component
 *
 * Renders Mermaid diagram code as SVG using the Mermaid library.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Copy, Download, Maximize2, Minimize2, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

interface MermaidDiagramProps {
  code: string;
  title?: string;
  className?: string;
}

// Declare mermaid on window for TypeScript
declare global {
  interface Window {
    mermaid: any;
  }
}

// Load Mermaid from CDN
const loadMermaid = (): Promise<void> => {
  return new Promise((resolve, reject) => {
    if (window.mermaid) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
    script.async = true;
    script.onload = () => {
      window.mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        securityLevel: 'loose',
        flowchart: {
          useMaxWidth: true,
          htmlLabels: true,
          curve: 'basis',
        },
      });
      resolve();
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

const MermaidDiagram: React.FC<MermaidDiagramProps> = ({
  code,
  title,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    let mounted = true;

    const renderDiagram = async () => {
      if (!code) {
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        setError('');

        await loadMermaid();

        if (!mounted) return;

        // Generate unique ID for this diagram
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;

        // Render the diagram
        const { svg: renderedSvg } = await window.mermaid.render(id, code);

        if (mounted) {
          setSvg(renderedSvg);
          setIsLoading(false);
        }
      } catch (err: any) {
        console.error('Mermaid render error:', err);
        if (mounted) {
          setError(err.message || 'Failed to render diagram');
          setIsLoading(false);
        }
      }
    };

    renderDiagram();

    return () => {
      mounted = false;
    };
  }, [code]);

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    toast.success('Mermaid code copied to clipboard');
  };

  const downloadSvg = () => {
    if (!svg) return;

    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title || 'diagram'}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Diagram downloaded');
  };

  if (isLoading) {
    return (
      <div className={`bg-white rounded-lg border p-4 ${className}`}>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          <span className="ml-2 text-gray-500">Rendering diagram...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white rounded-lg border p-4 ${className}`}>
        <div className="flex items-start space-x-2 text-red-600">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Failed to render diagram</p>
            <p className="text-sm text-red-500">{error}</p>
            <details className="mt-2">
              <summary className="text-xs text-gray-500 cursor-pointer">
                Show Mermaid code
              </summary>
              <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-x-auto">
                {code}
              </pre>
            </details>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg border ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b bg-gray-50 rounded-t-lg">
        <span className="text-sm font-medium text-gray-700">
          {title || 'Generated Diagram'}
        </span>
        <div className="flex items-center space-x-1">
          <button
            onClick={copyCode}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded"
            title="Copy Mermaid code"
          >
            <Copy className="w-4 h-4" />
          </button>
          <button
            onClick={downloadSvg}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded"
            title="Download as SVG"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Diagram */}
      <div
        ref={containerRef}
        className={`p-4 overflow-auto ${
          isExpanded ? 'max-h-[80vh]' : 'max-h-96'
        }`}
        dangerouslySetInnerHTML={{ __html: svg }}
      />

      {/* Code preview (collapsed by default) */}
      <details className="border-t">
        <summary className="px-3 py-2 text-xs text-gray-500 cursor-pointer hover:bg-gray-50">
          View Mermaid code
        </summary>
        <pre className="px-3 py-2 bg-gray-50 text-xs overflow-x-auto border-t">
          {code}
        </pre>
      </details>
    </div>
  );
};

export default MermaidDiagram;
