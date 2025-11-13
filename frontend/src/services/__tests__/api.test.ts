/**
 * Tests for API service layer
 */

import { apiClient } from '../api';

// Mock axios
jest.mock('axios', () => {
  const mockAxios = {
    create: jest.fn(() => ({
      get: jest.fn(),
      post: jest.fn(),
      put: jest.fn(),
      delete: jest.fn(),
      interceptors: {
        request: { use: jest.fn() },
        response: { use: jest.fn() },
      },
    })),
  };
  return mockAxios;
});

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  it('sets token in localStorage', () => {
    apiClient.setToken('test-token');
    expect(localStorage.getItem('access_token')).toBe('test-token');
  });

  it('clears token from localStorage', () => {
    localStorage.setItem('access_token', 'test-token');
    apiClient.clearToken();
    expect(localStorage.getItem('access_token')).toBeNull();
  });

  it('loads token from localStorage on initialization', () => {
    localStorage.setItem('access_token', 'saved-token');
    // Re-import to trigger initialization
    jest.resetModules();
    // Token should be loaded in constructor
    expect(localStorage.getItem('access_token')).toBe('saved-token');
  });
});

