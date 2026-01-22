/**
 * Visual input mapper component for workflow tool nodes.
 *
 * Displays tool parameters with their types and provides
 * autocomplete for context variable mappings.
 */

import React from 'react';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';
import ContextAutocomplete, { ContextVariable } from './ContextAutocomplete';

// Tool parameter definition
export interface ToolParameter {
  name: string;
  type: string;
  description?: string;
  required: boolean;
  default?: any;
  enum?: string[];
}

interface InputMapperProps {
  parameters: ToolParameter[];
  inputMapping: Record<string, string>;
  onChange: (mapping: Record<string, string>) => void;
  availableContext: ContextVariable[];
  errors?: Record<string, string>;
}

const InputMapper: React.FC<InputMapperProps> = ({
  parameters,
  inputMapping,
  onChange,
  availableContext,
  errors = {},
}) => {
  // Handle change for a single parameter
  const handleParameterChange = (paramName: string, value: string) => {
    const newMapping = { ...inputMapping };
    if (value) {
      newMapping[paramName] = value;
    } else {
      delete newMapping[paramName];
    }
    onChange(newMapping);
  };

  // Get type badge color
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'string':
        return 'bg-green-100 text-green-700';
      case 'integer':
        return 'bg-blue-100 text-blue-700';
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

  // Check if parameter has a value
  const hasValue = (paramName: string) => {
    return paramName in inputMapping && inputMapping[paramName] !== '';
  };

  // Render parameter input based on type
  const renderParameterInput = (param: ToolParameter) => {
    const value = inputMapping[param.name] || '';
    const error = errors[param.name];

    // For enum types, show a select dropdown
    if (param.enum && param.enum.length > 0) {
      return (
        <select
          value={value}
          onChange={(e) => handleParameterChange(param.name, e.target.value)}
          className={`w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 ${
            error ? 'border-red-300' : ''
          }`}
        >
          <option value="">Select...</option>
          {param.enum.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      );
    }

    // For boolean types, show a checkbox or select
    if (param.type === 'boolean') {
      return (
        <select
          value={value}
          onChange={(e) => handleParameterChange(param.name, e.target.value)}
          className={`w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 ${
            error ? 'border-red-300' : ''
          }`}
        >
          <option value="">Use context or default</option>
          <option value="true">true</option>
          <option value="false">false</option>
          <option value="{{context.variable}}">{"{{context.variable}}"}</option>
        </select>
      );
    }

    // For other types, show text input with autocomplete
    return (
      <ContextAutocomplete
        value={value}
        onChange={(newValue) => handleParameterChange(param.name, newValue)}
        availableVariables={availableContext}
        placeholder={
          param.default !== undefined
            ? `Default: ${JSON.stringify(param.default)}`
            : param.required
            ? 'Required'
            : 'Optional'
        }
        className={error ? 'border-red-300' : ''}
      />
    );
  };

  if (parameters.length === 0) {
    return (
      <div className="p-3 text-sm text-gray-500 text-center bg-gray-50 rounded border border-dashed">
        <Info className="w-4 h-4 mx-auto mb-1" />
        No parameters required for this tool
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {parameters.map((param) => {
        const error = errors[param.name];
        const isMapped = hasValue(param.name);

        return (
          <div key={param.name} className="space-y-1">
            {/* Parameter header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium text-gray-700">
                  {param.name}
                </label>
                {param.required && (
                  <span className="px-1 py-0.5 text-xs font-medium bg-red-100 text-red-600 rounded">
                    Required
                  </span>
                )}
                <span
                  className={`px-1.5 py-0.5 text-xs rounded ${getTypeColor(param.type)}`}
                >
                  {param.type}
                </span>
              </div>
              {/* Status indicator */}
              <div className="flex items-center">
                {error ? (
                  <AlertCircle className="w-4 h-4 text-red-500" />
                ) : isMapped ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : param.required ? (
                  <AlertCircle className="w-4 h-4 text-amber-500" />
                ) : null}
              </div>
            </div>

            {/* Description */}
            {param.description && (
              <p className="text-xs text-gray-500">{param.description}</p>
            )}

            {/* Input */}
            {renderParameterInput(param)}

            {/* Error message */}
            {error && (
              <p className="text-xs text-red-500 flex items-center">
                <AlertCircle className="w-3 h-3 mr-1" />
                {error}
              </p>
            )}
          </div>
        );
      })}

      {/* Unmapped required parameters warning */}
      {parameters.some((p) => p.required && !hasValue(p.name)) && (
        <div className="p-2 text-xs text-amber-700 bg-amber-50 rounded border border-amber-200 flex items-start">
          <AlertCircle className="w-3 h-3 mr-1 mt-0.5 flex-shrink-0" />
          <span>
            Some required parameters are not mapped. The workflow may fail if these
            values cannot be resolved from context.
          </span>
        </div>
      )}
    </div>
  );
};

export default InputMapper;
