/**
 * Modal for editing knowledge graph relationships
 */

import React, { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { X, ArrowRight, Save } from 'lucide-react';
import { apiClient } from '../../services/api';
import { KGRelationshipUpdate } from '../../types';
import toast from 'react-hot-toast';

// Standard relation types
const RELATION_TYPES = [
  'works_for',
  'manages',
  'reports_to',
  'collaborates_with',
  'owns',
  'uses',
  'implements',
  'part_of',
  'located_in',
  'related_to',
  'mentions',
  'references',
  'created_by',
];

interface RelationshipEditModalProps {
  relationshipId: string;
  onClose: () => void;
  onSaved?: () => void;
}

const RelationshipEditModal: React.FC<RelationshipEditModalProps> = ({
  relationshipId,
  onClose,
  onSaved,
}) => {
  const queryClient = useQueryClient();
  const [formData, setFormData] = useState<KGRelationshipUpdate>({});
  const [customType, setCustomType] = useState('');

  // Fetch relationship details
  const { data: relationship, isLoading } = useQuery(
    ['kgRelationship', relationshipId],
    () => apiClient.getKGRelationship(relationshipId),
    {
      onSuccess: (data) => {
        setFormData({
          relation_type: data.relation_type,
          confidence: data.confidence,
          evidence: data.evidence || '',
        });
        if (!RELATION_TYPES.includes(data.relation_type)) {
          setCustomType(data.relation_type);
        }
      },
    }
  );

  // Fetch available relation types from the system
  const { data: relationTypesData } = useQuery(
    ['kgRelationTypes'],
    () => apiClient.listKGRelationTypes()
  );

  const allRelationTypes = React.useMemo(() => {
    const systemTypes = relationTypesData?.types || [];
    const combined = Array.from(new Set([...RELATION_TYPES, ...systemTypes]));
    return combined.sort();
  }, [relationTypesData]);

  // Update mutation
  const updateMutation = useMutation(
    (data: KGRelationshipUpdate) => apiClient.updateKGRelationship(relationshipId, data),
    {
      onSuccess: () => {
        toast.success('Relationship updated');
        queryClient.invalidateQueries(['kgRelationship', relationshipId]);
        queryClient.invalidateQueries('kg-document-graph');
        queryClient.invalidateQueries('globalKGGraph');
        onSaved?.();
        onClose();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || 'Failed to update relationship';
        toast.error(message);
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const updates: KGRelationshipUpdate = {};

    // Only include changed fields
    if (formData.relation_type !== relationship?.relation_type) {
      updates.relation_type = formData.relation_type === 'custom' ? customType : formData.relation_type;
    }
    if (formData.confidence !== relationship?.confidence) {
      updates.confidence = formData.confidence;
    }
    if (formData.evidence !== (relationship?.evidence || '')) {
      updates.evidence = formData.evidence;
    }

    if (Object.keys(updates).length === 0) {
      toast('No changes to save');
      return;
    }

    updateMutation.mutate(updates);
  };

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 w-full max-w-lg">
          <div className="text-center text-gray-500">Loading...</div>
        </div>
      </div>
    );
  }

  if (!relationship) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">Edit Relationship</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Entity Display */}
        <div className="p-4 bg-gray-50 border-b">
          <div className="flex items-center justify-center gap-4">
            <div className="text-center">
              <div className="text-sm text-gray-500">Source</div>
              <div className="font-medium text-gray-900">{relationship.source_entity_name}</div>
            </div>
            <ArrowRight className="w-5 h-5 text-gray-400" />
            <div className="text-center">
              <div className="text-sm text-gray-500">Target</div>
              <div className="font-medium text-gray-900">{relationship.target_entity_name}</div>
            </div>
          </div>
          {relationship.is_manual && (
            <div className="mt-2 text-center">
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                Manual Relationship
              </span>
            </div>
          )}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Relation Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Relationship Type
            </label>
            <select
              value={RELATION_TYPES.includes(formData.relation_type || '') ? formData.relation_type : 'custom'}
              onChange={(e) => {
                if (e.target.value === 'custom') {
                  setFormData({ ...formData, relation_type: customType || '' });
                } else {
                  setFormData({ ...formData, relation_type: e.target.value });
                }
              }}
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              {allRelationTypes.map((type) => (
                <option key={type} value={type}>
                  {type.replace(/_/g, ' ')}
                </option>
              ))}
              <option value="custom">Custom...</option>
            </select>
            {(!RELATION_TYPES.includes(formData.relation_type || '') || formData.relation_type === 'custom') && (
              <input
                type="text"
                value={customType}
                onChange={(e) => {
                  const val = e.target.value.toLowerCase().replace(/\s+/g, '_');
                  setCustomType(val);
                  setFormData({ ...formData, relation_type: val });
                }}
                placeholder="Enter custom type (e.g., authored_by)"
                className="mt-2 w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              />
            )}
          </div>

          {/* Confidence */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Confidence: {((formData.confidence || 0) * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={(formData.confidence || 0) * 100}
              onChange={(e) => setFormData({ ...formData, confidence: Number(e.target.value) / 100 })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>

          {/* Evidence */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Evidence (optional)
            </label>
            <textarea
              value={formData.evidence || ''}
              onChange={(e) => setFormData({ ...formData, evidence: e.target.value })}
              rows={3}
              placeholder="Supporting text or explanation for this relationship..."
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            />
          </div>

          {/* Metadata */}
          {relationship.document_id && (
            <div className="text-xs text-gray-500">
              <p>Extracted from document (not manual)</p>
              {relationship.chunk_id && <p>Chunk ID: {relationship.chunk_id}</p>}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={updateMutation.isLoading}
              className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              {updateMutation.isLoading ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RelationshipEditModal;
