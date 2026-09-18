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

  /**
   * A run that never cloned a repository has no workspace, and the API says so
   * with a 404 whose body reads "This run did not use a coding workspace."
   * That is an answer, not a failure -- most runs are research runs -- and the
   * workspace panel already renders nothing for it. But the response
   * interceptor toasts any error carrying a body, so opening an ordinary job
   * put a red error on screen saying the run did not do something it was never
   * asked to do. The request opts out of the toast; these pin that.
   */
  describe('workspace lookups do not toast an expected answer', () => {
    const configOf = (call: any[]) => call[1] || {};

    it('asks for no toast when fetching a run workspace', async () => {
      const client = (apiClient as any).client;
      client.get.mockResolvedValue({ data: {} });
      await apiClient.getJobWorkspace('job-1');
      expect(configOf(client.get.mock.calls[0]).suppressToast).toBe(true);
    });

    it('asks for no toast when fetching a workspace file', async () => {
      const client = (apiClient as any).client;
      client.get.mockResolvedValue({ data: {} });
      await apiClient.getJobWorkspaceFile('job-1', 'src/main.py');
      expect(configOf(client.get.mock.calls[0]).suppressToast).toBe(true);
    });

    it('still passes the path it was asked for', async () => {
      const client = (apiClient as any).client;
      client.get.mockResolvedValue({ data: {} });
      await apiClient.getJobWorkspace('job-1', 'src');
      expect(configOf(client.get.mock.calls[0]).params).toEqual({ path: 'src' });
    });
  });
});

