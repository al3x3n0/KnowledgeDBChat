import * as React from 'react';

export function useElementSize<T extends HTMLElement>() {
  const ref = React.useRef<T | null>(null);
  const [size, setSize] = React.useState({ width: 0, height: 0 });

  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const cr = entry.contentRect;
      setSize({ width: Math.round(cr.width), height: Math.round(cr.height) });
    });

    ro.observe(el);
    // Prime immediately.
    const r = el.getBoundingClientRect();
    setSize({ width: Math.round(r.width), height: Math.round(r.height) });

    return () => ro.disconnect();
  }, []);

  return { ref, ...size };
}

