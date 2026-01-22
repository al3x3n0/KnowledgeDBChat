/**
 * Context variable autocomplete input component.
 *
 * Provides autocomplete suggestions for {{context.xxx}} expressions
 * in workflow input mappings.
 */

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { ChevronDown, Variable, Info } from 'lucide-react';

// Context variable type
export interface ContextVariable {
  path: string;
  type: string;
  from_node: string;
  description?: string;
}

interface ContextAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  availableVariables: ContextVariable[];
  placeholder?: string;
  className?: string;
}

const ContextAutocomplete: React.FC<ContextAutocompleteProps> = ({
  value,
  onChange,
  availableVariables,
  placeholder = 'Enter value or {{context.path}}',
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Filter variables based on input
  const filteredVariables = useMemo(() => {
    if (!filter) return availableVariables;
    const lowerFilter = filter.toLowerCase();
    return availableVariables.filter(
      (v) =>
        v.path.toLowerCase().includes(lowerFilter) ||
        v.from_node.toLowerCase().includes(lowerFilter) ||
        v.description?.toLowerCase().includes(lowerFilter)
    );
  }, [availableVariables, filter]);

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    onChange(newValue);

    // Check if user is typing a template expression
    if (newValue.includes('{{') && !newValue.includes('}}')) {
      setIsOpen(true);
      // Extract partial path for filtering
      const match = newValue.match(/\{\{(.*)$/);
      if (match) {
        setFilter(match[1].trim());
      }
    } else {
      setFilter('');
    }
  };

  // Handle variable selection
  const handleSelect = (variable: ContextVariable) => {
    // Replace any partial {{...}} with the full expression
    const baseValue = value.replace(/\{\{[^}]*$/, '');
    onChange(baseValue + `{{${variable.path}}}`);
    setIsOpen(false);
    setFilter('');
    inputRef.current?.focus();
  };

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Get type badge color
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'string':
        return 'bg-green-100 text-green-700';
      case 'integer':
      case 'number':
        return 'bg-blue-100 text-blue-700';
      case 'boolean':
        return 'bg-purple-100 text-purple-700';
      case 'array':
        return 'bg-orange-100 text-orange-700';
      case 'object':
        return 'bg-gray-100 text-gray-700';
      default:
        return 'bg-gray-100 text-gray-600';
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div className="flex">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onFocus={() => {
            if (value.includes('{{') && !value.includes('}}')) {
              setIsOpen(true);
            }
          }}
          placeholder={placeholder}
          className="flex-1 px-2 py-1 text-sm border rounded-l focus:ring-1 focus:ring-primary-500 focus:border-primary-500 font-mono"
        />
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="px-2 py-1 border border-l-0 rounded-r bg-gray-50 hover:bg-gray-100"
          title="Show available context variables"
        >
          <Variable className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      {/* Dropdown */}
      {isOpen && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-white border rounded-md shadow-lg max-h-60 overflow-y-auto"
        >
          {/* Search filter */}
          <div className="p-2 border-b sticky top-0 bg-white">
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter variables..."
              className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500"
              autoFocus
            />
          </div>

          {/* Variables list */}
          {filteredVariables.length > 0 ? (
            <ul className="py-1">
              {filteredVariables.map((variable, index) => (
                <li key={`${variable.path}-${index}`}>
                  <button
                    type="button"
                    onClick={() => handleSelect(variable)}
                    className="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none"
                  >
                    <div className="flex items-center justify-between">
                      <code className="text-sm text-gray-800 font-mono">
                        {`{{${variable.path}}}`}
                      </code>
                      <span
                        className={`px-1.5 py-0.5 text-xs rounded ${getTypeColor(variable.type)}`}
                      >
                        {variable.type}
                      </span>
                    </div>
                    <div className="flex items-center mt-1 text-xs text-gray-500">
                      <span className="truncate">
                        From: <span className="font-medium">{variable.from_node}</span>
                      </span>
                      {variable.description && (
                        <>
                          <span className="mx-1">•</span>
                          <span className="truncate">{variable.description}</span>
                        </>
                      )}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="p-4 text-sm text-gray-500 text-center">
              No context variables available
              {filter && ` matching "${filter}"`}
            </div>
          )}

          {/* Help text */}
          <div className="p-2 border-t bg-gray-50 text-xs text-gray-500 flex items-start">
            <Info className="w-3 h-3 mr-1 mt-0.5 flex-shrink-0" />
            <span>
              Type <code className="bg-gray-200 px-1 rounded">{'{{'}</code> to insert a context variable.
              Variables come from outputs of upstream nodes.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ContextAutocomplete;
