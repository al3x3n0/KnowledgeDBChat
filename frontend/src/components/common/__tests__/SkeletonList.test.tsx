/**
 * The loading placeholder, and the two things about it that are not cosmetic.
 *
 * A skeleton is decoration that a screen reader can easily turn into noise —
 * a dozen empty list items announced one by one — so what it exposes to
 * assistive technology matters more than how it looks.
 */

import { render, screen } from '@testing-library/react';
import React from 'react';

import SkeletonList from '../SkeletonList';

it('announces what is loading, once', () => {
  render(<SkeletonList rows={5} label="Loading documents" />);

  const status = screen.getByRole('status');
  expect(status).toHaveAttribute('aria-busy', 'true');
  expect(status).toHaveAttribute('aria-live', 'polite');
  // One announcement for the whole list, not one per row.
  expect(screen.getByText('Loading documents')).toBeInTheDocument();
  expect(screen.getAllByText('Loading documents')).toHaveLength(1);
});

it('hides the bars themselves from assistive technology', () => {
  const { container } = render(<SkeletonList rows={4} />);

  const rows = container.querySelectorAll('[aria-hidden="true"]');
  expect(rows).toHaveLength(4);
});

it('draws the number of rows asked for', () => {
  const { container, rerender } = render(<SkeletonList rows={3} />);
  expect(container.querySelectorAll('[aria-hidden="true"]')).toHaveLength(3);

  rerender(<SkeletonList rows={7} />);
  expect(container.querySelectorAll('[aria-hidden="true"]')).toHaveLength(7);
});

it('staggers the shimmer on the animated elements, not their container', () => {
  // animation-delay is not inherited: a delay set on the row would do nothing,
  // and every bar would flash at once.
  const { container } = render(<SkeletonList rows={3} />);

  const shimmering = Array.from(container.querySelectorAll('.skeleton')) as HTMLElement[];
  expect(shimmering.length).toBeGreaterThan(0);
  expect(shimmering.every((el) => el.style.animationDelay !== '')).toBe(true);

  const delays = Array.from(container.querySelectorAll('[aria-hidden="true"]')).map((row) =>
    (row.querySelector('.skeleton') as HTMLElement).style.animationDelay
  );
  expect(delays).toEqual(['0ms', '90ms', '180ms']);
});

it('drops the card furniture in the dense row variant', () => {
  const { container } = render(<SkeletonList rows={2} variant="row" />);

  // No avatar block and no status pill: a dense row has neither.
  expect(container.querySelectorAll('.skeleton')).toHaveLength(4); // two bars per row
  expect(container.querySelector('.rounded-lg')).toBeNull();
});
