/**
 * Modal for creating new knowledge graph relationships
 */

import React, { useState, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { X, ArrowRight, Plus, Search } from 'lucide-react';
import { apiClient } from '../../services/api';
import { KGRelationshipCreate } from '../../types';
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

interface Entity {
  id: string;
  name: string;
  type: string;
}

interface RelationshipCreateModalProps {
  documentId?: string;
  onClose: () => void;
  onCreated?: () => void;
}

const RelationshipCreateModal: React.FC<RelationshipCreateModalProps> = ({
  documentId,
  onClose,
  onCreated,
}) => {
  const queryClient = useQueryClient();
  const [sourceEntity, setSourceEntity] = useState<Entity | null>(null);
  const [targetEntity, setTargetEntity] = useState<Entity | null>(null);
  const [relationType, setRelationType] = useState('related_to');
  const [customType, setCustomType] = useState('');
  const [confidence, setConfidence] = useState(0.8);
  const [evidence, setEvidence] = useState('');

  // Search states
  const [sourceSearch, setSourceSearch] = useState('');
  const [targetSearch, setTargetSearch] = useState('');
  const [activeSearch, setActiveSearch] = useState<'source' | 'target' | null>(null);

  // Fetch entities for the document or globally
  const { data: graph } = useQuery(
    documentId ? ['kg-document-graph', documentId] : ['kgGlobalGraph'],
    () => documentId
      ? apiClient.getKGDocumentGraph(documentId)
      : apiClient.getGlobalKGGraph({ limit_nodes: 1000 }),
    { staleTime: 60000 }
  );

  const entities: Entity[] = useMemo(() => {
    if (!graph?.nodes) return [];
    return graph.nodes.map((n: any) => ({
      id: n.id,
      name: n.name || n.label || n.id,
      type: n.type || 'unknown',
    }));
  }, [graph]);

  // Filter entities based on search
  const filteredEntities = useMemo(() => {
    const search = activeSearch === 'source' ? sourceSearch : targetSearch;
    if (!search.trim()) return entities.slice(0, 50);
    const lower = search.toLowerCase();
    return entities.filter(e =>
      e.name.toLowerCase().includes(lower) ||
      e.type.toLowerCase().includes(lower)
    ).slice(0, 50);
  }, [entities, activeSearch, sourceSearch, targetSearch]);

  // Fetch available relation types from the system
  const { data: relationTypesData } = useQuery(
    ['kgRelationTypes'],
    () => apiClient.listKGRelationTypes()
  );

  const allRelationTypes = useMemo(() => {
    const systemTypes = relationTypesData?.types || [];
    const combined = Array.from(new Set([...RELATION_TYPES, ...systemTypes]));
    return combined.sort();
  }, [relationTypesData]);

  // Create mutation
  const createMutation = useMutation(
    (data: KGRelationshipCreate) => apiClient.createKGRelationship(data),
    {
      onSuccess: () => {
        toast.success('Relationship created');
        queryClient.invalidateQueries('kg-document-graph');
        queryClient.invalidateQueries('globalKGGraph');
        onCreated?.();
        onClose();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || 'Failed to create relationship';
        toast.error(message);
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!sourceEntity) {
      toast.error('Please select a source entity');
      return;
    }
    if (!targetEntity) {
      toast.error('Please select a target entity');
      return;
    }
    if (sourceEntity.id === targetEntity.id) {
      toast.error('Source and target entities must be different');
      return;
    }

    const finalType = relationType === 'custom' ? customType : relationType;
    if (!finalType.trim()) {
      toast.error('Please specify a relationship type');
      return;
    }

    createMutation.mutate({
      source_entity_id: sourceEntity.id,
      target_entity_id: targetEntity.id,
      relation_type: finalType,
      confidence,
      evidence: evidence.trim() || undefined,
    });
  };

  const renderEntityPicker = (
    type: 'source' | 'target',
    selected: Entity | null,
    setSelected: (e: Entity | null) => void,
    search: string,
    setSearch: (s: string) => void
  ) => {
    const isActive = activeSearch === type;

    return (
      <div className="relative">
        <label className="block text-sm font-medium text-gray-700 mb-1 capitalize">
          {type} Entity
        </label>
        {selected ? (
          <div className="flex items-center justify-between p-2 border border-gray-300 rounded-lg bg-gray-50">
            <div>
              <span className="font-medium text-gray-900">{selected.name}</span>
              <span className="ml-2 text-xs text-gray-500 bg-gray-200 px-1.5 py-0.5 rounded">
                {selected.type}
              </span>
            </div>
            <button
              type="button"
              className="text-gray-500 hover:text-gray-700"
              onClick={() => setSelected(null)}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="relative">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onFocus={() => setActiveSearch(type)}
                placeholder={`Search ${type} entity...`}
                className="w-full pl-9 pr-3 py-2 rounded-lg border border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              />
            </div>
            {isActive && (
              <div className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-48 overflow-auto">
                {filteredEntities.length === 0 ? (
                  <div className="p-3 text-sm text-gray-500 text-center">No entities found</div>
                ) : (
                  filteredEntities.map((entity) => (
                    <button
                      key={entity.id}
                      type="button"
                      className="w-full text-left px-3 py-2 hover:bg-gray-50 border-b border-gray-100 last:border-b-0"
                      onClick={() => {
                        setSelected(entity);
                        setSearch('');
                        setActiveSearch(null);
                      }}
                    >
                      <span className="font-medium text-gray-900">{entity.name}</span>
                      <span className="ml-2 text-xs text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded">
                        {entity.type}
                      </span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setActiveSearch(null)}>
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">Create Relationship</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Source Entity */}
          {renderEntityPicker('source', sourceEntity, setSourceEntity, sourceSearch, setSourceSearch)}

          {/* Arrow indicator */}
          {sourceEntity && targetEntity && (
            <div className="flex items-center justify-center py-1">
              <div className="flex items-center gap-2 text-gray-400">
                <span className="text-sm">{sourceEntity.name}</span>
                <ArrowRight className="w-4 h-4" />
                <span className="text-sm">{targetEntity.name}</span>
              </div>
            </div>
          )}

          {/* Target Entity */}
          {renderEntityPicker('target', targetEntity, setTargetEntity, targetSearch, setTargetSearch)}

          {/* Relation Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Relationship Type
            </label>
            <select
              value={RELATION_TYPES.includes(relationType) ? relationType : 'custom'}
              onChange={(e) => {
                if (e.target.value === 'custom') {
                  setRelationType('custom');
                } else {
                  setRelationType(e.target.value);
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
            {relationType === 'custom' && (
              <input
                type="text"
                value={customType}
                onChange={(e) => {
                  const val = e.target.value.toLowerCase().replace(/\s+/g, '_');
                  setCustomType(val);
                }}
                placeholder="Enter custom type (e.g., authored_by)"
                className="mt-2 w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              />
            )}
          </div>

          {/* Confidence */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Confidence: {(confidence * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={confidence * 100}
              onChange={(e) => setConfidence(Number(e.target.value) / 100)}
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
              value={evidence}
              onChange={(e) => setEvidence(e.target.value)}
              rows={3}
              placeholder="Supporting text or explanation for this relationship..."
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            />
          </div>

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
              disabled={createMutation.isLoading || !sourceEntity || !targetEntity}
              className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              {createMutation.isLoading ? 'Creating...' : 'Create Relationship'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RelationshipCreateModal;
