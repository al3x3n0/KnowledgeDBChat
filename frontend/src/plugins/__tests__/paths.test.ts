import { asCell, asRows, resolvePath } from '../paths';

describe('resolvePath', () => {
  it('returns the whole payload for an empty path', () => {
    const data = { a: 1 };
    expect(resolvePath(data, '')).toEqual({ found: true, value: data });
  });

  it('walks a dotted path', () => {
    expect(resolvePath({ a: { b: [1, 2] } }, 'a.b')).toEqual({
      found: true,
      value: [1, 2],
    });
  });

  it('says where a wrong path broke', () => {
    // The whole reason this is separate and pure: `result.items` against a
    // tool returning `{items: [...]}` must be diagnosable, not a blank table.
    expect(resolvePath({ items: [] }, 'result.items')).toEqual({
      found: false,
      missingAt: 'result',
    });
  });

  it('distinguishes a missing path from a legitimately empty value', () => {
    expect(resolvePath({ items: [] }, 'items')).toEqual({
      found: true,
      value: [],
    });
  });

  it('stops when the path runs into a scalar', () => {
    expect(resolvePath({ a: 5 }, 'a.b')).toEqual({ found: false, missingAt: 'a' });
  });

  it('treats a null payload as nothing found', () => {
    expect(resolvePath(null, 'a')).toEqual({ found: false, missingAt: '(root)' });
  });
});

describe('asRows', () => {
  it('passes an array of objects through', () => {
    expect(asRows([{ a: 1 }, { b: 2 }])).toHaveLength(2);
  });

  it('drops entries that are not objects', () => {
    expect(asRows([{ a: 1 }, 'nope', 3, null])).toEqual([{ a: 1 }]);
  });

  it('treats a lone object as one row', () => {
    // Tools returning "the latest" rather than "the list" are common enough
    // that refusing them would be pedantry.
    expect(asRows({ a: 1 })).toEqual([{ a: 1 }]);
  });

  it('has no rows for a scalar', () => {
    expect(asRows('text')).toEqual([]);
    expect(asRows(null)).toEqual([]);
  });
});

describe('asCell', () => {
  it('renders scalars as themselves', () => {
    expect(asCell('x')).toBe('x');
    expect(asCell(31)).toBe('31');
    expect(asCell(false)).toBe('false');
  });

  it('marks absence rather than printing undefined', () => {
    expect(asCell(null)).toBe('—');
    expect(asCell(undefined)).toBe('—');
  });

  it('renders an object as JSON, not [object Object]', () => {
    // The difference between a table you can debug and one you cannot.
    expect(asCell({ a: 1 })).toBe('{"a":1}');
  });

  it('survives a value that cannot be serialized', () => {
    const cyclic: any = {};
    cyclic.self = cyclic;
    expect(() => asCell(cyclic)).not.toThrow();
  });
});
