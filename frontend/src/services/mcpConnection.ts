interface ResourceSummary {
  uri: string;
  name: string;
}

interface PromptSummary {
  id: string;
  name: string;
}

type Discovery<T> =
  | { status: 'unavailable' }
  | { status: 'listed'; items: T[] }
  | { status: 'failed'; message: string };

export interface MCPConnectionResult {
  protocolVersion: string;
  tools: string[];
  resources: Discovery<ResourceSummary>;
  prompts: Discovery<PromptSummary>;
}

class MCPConnectionError extends Error {}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export async function checkMCPConnection(endpoint: string, apiKey: string): Promise<MCPConnectionResult> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10000);
  const headers = {
    Accept: 'application/json, text/event-stream',
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };

  async function send(message: { method: string; id?: number; params?: object }, protocolVersion?: string) {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: protocolVersion ? { ...headers, 'MCP-Protocol-Version': protocolVersion } : headers,
      body: JSON.stringify({ jsonrpc: '2.0', ...message }),
      credentials: 'omit',
      redirect: 'error',
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) {
      const messages: Record<number, string> = {
        401: 'The API key is invalid, expired, or revoked.',
        403: 'MCP access is disabled or this dashboard origin is blocked.',
        404: 'The MCP endpoint could not be found.',
        405: 'The endpoint does not accept MCP requests.',
        429: 'The API key reached its request limit. Try again later.',
      };
      throw new MCPConnectionError(messages[response.status] ?? `The server returned HTTP ${response.status}.`);
    }
    return response;
  }

  async function result(response: Response, id: number) {
    const message = await response.json().catch((error) => {
      if (controller.signal.aborted) throw error;
      return null;
    });
    if (message?.jsonrpc !== '2.0' || message.id !== id || !isObject(message.result) || message.error) {
      throw new MCPConnectionError('The endpoint did not return a valid MCP response.');
    }
    return message.result;
  }

  async function discover<T>(
    feature: 'resources' | 'prompts',
    capabilities: unknown,
    protocolVersion: string,
    summarize: (entry: Record<string, unknown>, id: string, name: string) => T
  ): Promise<Discovery<T>> {
    if (!isObject(capabilities) || !isObject(capabilities[feature])) {
      return { status: 'unavailable' };
    }
    const id = feature === 'resources' ? 3 : 4;
    const label = feature === 'resources' ? 'resource' : 'prompt';
    try {
      const response = await result(await send({ id, method: `${feature}/list` }, protocolVersion), id);
      const entries = response[feature];
      if (!Array.isArray(entries)) {
        throw new MCPConnectionError(`The server did not return a valid ${label} list.`);
      }
      const items = entries.map((entry: unknown) => {
        if (!isObject(entry) || typeof entry.name !== 'string') {
          throw new MCPConnectionError(`The server did not return a valid ${label} list.`);
        }
        return summarize(entry, entry.name, typeof entry.title === 'string' ? entry.title : entry.name);
      });
      return { status: 'listed', items };
    } catch (error) {
      return {
        status: 'failed',
        message: error instanceof MCPConnectionError
          ? error.message
          : controller.signal.aborted
            ? 'The request timed out.'
            : 'The browser could not reach the endpoint.',
      };
    }
  }

  try {
    const handshake = await result(await send({
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2025-11-25',
        capabilities: {},
        clientInfo: { name: 'knowledgedb-connection-check', version: '1' },
      },
    }), 1);
    const { protocolVersion } = handshake;
    if (typeof protocolVersion !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(protocolVersion)) {
      throw new MCPConnectionError('The server did not select an MCP protocol version.');
    }
    const notification = await send({ method: 'notifications/initialized' }, protocolVersion);
    if (notification.status !== 202) {
      throw new MCPConnectionError('The server did not acknowledge MCP initialization.');
    }
    const { tools } = await result(await send({ id: 2, method: 'tools/list' }, protocolVersion), 2);
    if (!Array.isArray(tools) || tools.some((tool) => typeof tool?.name !== 'string')) {
      throw new MCPConnectionError('The server did not return a valid MCP tool list.');
    }
    const resources = await discover('resources', handshake.capabilities, protocolVersion, (entry, _id, name) => {
      if (typeof entry.uri !== 'string') {
        throw new MCPConnectionError('The server did not return a valid resource list.');
      }
      return { uri: entry.uri, name };
    });
    const prompts = await discover('prompts', handshake.capabilities, protocolVersion, (_entry, id, name) => ({ id, name }));
    return { protocolVersion, tools: tools.map(({ name }) => name), resources, prompts };
  } catch (error) {
    if (error instanceof MCPConnectionError) throw error;
    throw new MCPConnectionError(controller.signal.aborted
      ? 'The MCP connection timed out.'
      : 'The browser could not reach the MCP endpoint. Check the URL and origin settings.');
  } finally {
    window.clearTimeout(timeout);
  }
}
