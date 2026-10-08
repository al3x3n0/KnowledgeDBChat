import { readFileSync } from 'fs';
import { join } from 'path';

import { MERMAID_INTEGRITY, MERMAID_SRC } from '../MermaidDiagram';

const source = readFileSync(join(__dirname, '..', 'MermaidDiagram.tsx'), 'utf8');

describe('MermaidDiagram', () => {
  it('renders in strict mode', () => {
    // The diagram source is model output over ingested documents, and the SVG
    // is injected as HTML. 'loose' allows HTML labels and click handlers.
    expect(source).toContain("securityLevel: 'strict'");
    expect(source).not.toContain("securityLevel: 'loose'");
  });

  it('loads an exact version with an integrity hash', () => {
    expect(MERMAID_SRC).toMatch(/mermaid@\d+\.\d+\.\d+\/dist\/mermaid\.min\.js$/);
    expect(MERMAID_INTEGRITY).toMatch(/^sha384-[A-Za-z0-9+/]{64}$/);
    expect(source).toContain('script.integrity = MERMAID_INTEGRITY');
    expect(source).toContain("script.crossOrigin = 'anonymous'");
  });
});
