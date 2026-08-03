import { act, renderHook } from '@testing-library/react';

import { useAgentWebSocket } from '../useAgentWebSocket';

class MockAgentWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockAgentWebSocket[] = [];

  readonly url: string;
  readyState = MockAgentWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  send = jest.fn();

  constructor(url: string) {
    this.url = url;
    MockAgentWebSocket.instances.push(this);
  }

  receive(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  close() {
    if (this.readyState === MockAgentWebSocket.CLOSED) return;
    this.readyState = MockAgentWebSocket.CLOSED;
    this.onclose?.();
  }
}

describe('useAgentWebSocket', () => {
  const originalFetch = global.fetch;
  const originalWebSocket = global.WebSocket;

  beforeEach(() => {
    jest.useFakeTimers();
    MockAgentWebSocket.instances = [];
    global.WebSocket = MockAgentWebSocket as unknown as typeof WebSocket;
    global.fetch = jest.fn(() => new Promise<Response>(() => {}));
    localStorage.setItem('access_token', 'test-token');
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    global.fetch = originalFetch;
    global.WebSocket = originalWebSocket;
    localStorage.removeItem('access_token');
  });

  it('routes incoming websocket events through the initialized message handler', () => {
    const { result, unmount } = renderHook(() => useAgentWebSocket());
    const socket = MockAgentWebSocket.instances[0];

    act(() => socket.receive({ type: 'thinking', message: 'Reviewing documents' }));

    expect(result.current.streamingState).toEqual({
      phase: 'thinking',
      message: 'Reviewing documents',
      completedTools: [],
    });
    unmount();
  });

  it('does not reconnect after an intentional disconnect', () => {
    const { result, unmount } = renderHook(() => useAgentWebSocket());

    act(() => result.current.disconnect());
    act(() => jest.advanceTimersByTime(3000));

    expect(MockAgentWebSocket.instances).toHaveLength(1);
    expect(result.current.connectionStatus).toBe('disconnected');
    unmount();
  });

  it('reconnects after an unexpected close', () => {
    const { unmount } = renderHook(() => useAgentWebSocket());
    const socket = MockAgentWebSocket.instances[0];

    act(() => socket.close());
    act(() => jest.advanceTimersByTime(3000));

    expect(MockAgentWebSocket.instances).toHaveLength(2);
    expect(MockAgentWebSocket.instances[1].url).toContain('/api/v1/agent/ws?token=test-token');
    unmount();
  });
});
