import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';

import { safeExternalUrl } from '../safeUrl';

const SRC = join(__dirname, '..', '..');

describe('safeExternalUrl', () => {
  it('lets http(s) and same-origin paths through unchanged', () => {
    expect(safeExternalUrl('https://arxiv.org/abs/1805.10941')).toBe(
      'https://arxiv.org/abs/1805.10941'
    );
    expect(safeExternalUrl(' http://example.com/a?b=1 ')).toBe('http://example.com/a?b=1');
    expect(safeExternalUrl('/api/v1/documents/1/download')).toBe('/api/v1/documents/1/download');
  });

  it.each([
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    ' javascript:alert(1)',
    'data:text/html,<script>alert(1)</script>',
    'vbscript:msgbox(1)',
    'file:///etc/passwd',
    '//evil.example.com/x',
    'not a url',
    '',
  ])('refuses %s', (url) => {
    expect(safeExternalUrl(url)).toBeUndefined();
  });

  it('refuses what is not a string', () => {
    expect(safeExternalUrl(undefined)).toBeUndefined();
    expect(safeExternalUrl(null)).toBeUndefined();
    expect(safeExternalUrl({ href: 'https://x' })).toBeUndefined();
  });
});

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      return name === '__tests__' || name === 'node_modules' ? [] : sources(path);
    }
    return path.endsWith('.tsx') ? [path] : [];
  });
}

// The whole opening tag around each `target="_blank"`.
function newTabTags(text: string): string[] {
  const tags: string[] = [];
  let at = text.indexOf('target="_blank"');
  while (at !== -1) {
    const start = text.lastIndexOf('<', at);
    let end = text.indexOf('>', at);
    while (end !== -1 && text[end - 1] === '=') end = text.indexOf('>', end + 1);
    tags.push(text.slice(start, end + 1));
    at = text.indexOf('target="_blank"', at + 1);
  }
  return tags;
}

describe('links that open a new tab', () => {
  const files = sources(SRC);

  it('found the pages it is meant to check', () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it('all say rel="noopener"', () => {
    const missing = files.flatMap((file) =>
      newTabTags(readFileSync(file, 'utf8'))
        .filter((tag) => !tag.includes('noopener'))
        .map(() => file.replace(SRC, ''))
    );
    expect(missing).toEqual([]);
  });

  it('never take their href straight from a field of the data', () => {
    // `href={paper.pdf_url}` is somebody else's text as a link. A template
    // literal that starts with our own scheme or path, or a value built by
    // this code, is ours.
    const raw = /href=\{\s*[A-Za-z_$][\w$]*(\.[\w$]+)+\s*\}/;
    const offenders = files.flatMap((file) =>
      newTabTags(readFileSync(file, 'utf8'))
        .filter((tag) => raw.test(tag))
        .map((tag) => `${file.replace(SRC, '')}: ${tag.match(raw)?.[0]}`)
    );
    expect(offenders).toEqual([]);
  });
});
