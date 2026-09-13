import { act, renderHook } from '@testing-library/react';

import { useWebSocket } from '../useWebSocket';

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: ((error: Event) => void) | null = null;
  send = jest.fn();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  receive(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  close() {
    if (this.readyState === MockWebSocket.CLOSED) return;
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

describe('useWebSocket', () => {
  const originalWebSocket = global.WebSocket;

  beforeEach(() => {
    jest.useFakeTimers();
    MockWebSocket.instances = [];
    global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    global.WebSocket = originalWebSocket;
  });

  it('connects automatically and sends serialized messages once open', () => {
    const onOpen = jest.fn();
    const { result } = renderHook(() => useWebSocket('ws://example.test/socket', { onOpen }));

    expect(MockWebSocket.instances).toHaveLength(1);
    const socket = MockWebSocket.instances[0];

    act(() => socket.open());

    expect(result.current.connectionStatus).toBe('connected');
    expect(onOpen).toHaveBeenCalledTimes(1);

    act(() => {
      expect(result.current.sendMessage({ type: 'ping' })).toBe(true);
    });
    expect(socket.send).toHaveBeenCalledWith('{"type":"ping"}');
  });

  it('delivers messages to the latest callback without reconnecting', () => {
    const firstOnMessage = jest.fn();
    const latestOnMessage = jest.fn();
    const { result, rerender } = renderHook(
      ({ onMessage }) => useWebSocket('ws://example.test/socket', { onMessage }),
      { initialProps: { onMessage: firstOnMessage } }
    );
    const socket = MockWebSocket.instances[0];

    rerender({ onMessage: latestOnMessage });
    act(() => socket.receive({ type: 'notification', id: 'message-1' }));

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(firstOnMessage).not.toHaveBeenCalled();
    expect(latestOnMessage).toHaveBeenCalledWith({
      type: 'notification',
      id: 'message-1',
    });
    expect(result.current.lastMessage).toEqual({
      type: 'notification',
      id: 'message-1',
    });
  });

  it('does not reconnect after an intentional disconnect', () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://example.test/socket', {
        reconnectAttempts: 2,
        reconnectInterval: 1000,
      })
    );

    act(() => result.current.disconnect());
    act(() => jest.advanceTimersByTime(2000));

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(result.current.connectionStatus).toBe('disconnected');
  });

  it('reconnects after an unexpected close up to the configured policy', () => {
    renderHook(() =>
      useWebSocket('ws://example.test/socket', {
        reconnectAttempts: 2,
        reconnectInterval: 1000,
      })
    );
    const socket = MockWebSocket.instances[0];

    act(() => socket.close());
    expect(MockWebSocket.instances).toHaveLength(1);

    act(() => jest.advanceTimersByTime(1000));

    expect(MockWebSocket.instances).toHaveLength(2);
    expect(MockWebSocket.instances[1].url).toBe('ws://example.test/socket');
  });
});
