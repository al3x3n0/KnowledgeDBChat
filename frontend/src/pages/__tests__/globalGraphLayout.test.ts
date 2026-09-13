/**
 * How much of the canvas area the graph gets.
 *
 * The bug this covers: the details panel was rendered unconditionally at a
 * quarter of the width, so the canvas never filled the page even with nothing
 * selected — and the two span classes were hard-coded, so they had to agree
 * with the panels by hand.
 */

import { graphColumnSpanClass } from '../GlobalGraphPage';

it('gives the canvas everything when both panels are closed', () => {
  expect(graphColumnSpanClass(false, false)).toBe('lg:col-span-12');
});

it('leaves room for whichever single panel is open', () => {
  expect(graphColumnSpanClass(true, false)).toBe('lg:col-span-9');
  expect(graphColumnSpanClass(false, true)).toBe('lg:col-span-9');
});

it('splits the difference when both are open', () => {
  expect(graphColumnSpanClass(true, true)).toBe('lg:col-span-6');
});

it('always leaves the columns adding up to twelve', () => {
  // The property that matters: panels plus canvas fill the grid exactly, so
  // no column is ever left blank beside the graph.
  for (const filters of [false, true]) {
    for (const details of [false, true]) {
      const span = Number(graphColumnSpanClass(filters, details).replace('lg:col-span-', ''));
      const panels = (filters ? 3 : 0) + (details ? 3 : 0);
      expect(span + panels).toBe(12);
    }
  }
});
