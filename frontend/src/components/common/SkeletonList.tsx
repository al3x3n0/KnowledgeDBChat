/**
 * A placeholder for a list that is still arriving.
 *
 * A spinner says "wait"; a skeleton says "here is what is coming". The
 * difference matters most on this app's list pages, where the answer is
 * usually a dozen cards of roughly known shape — showing that shape makes the
 * page feel like it is filling in rather than blocked.
 *
 * Two rules keep it from becoming decoration in its own right:
 *
 * - It mimics the real content's geometry. A skeleton whose rows are a
 *   different height from the rows that replace them produces a visible jump
 *   at the moment the data lands, which is worse than the spinner it replaced.
 * - It is announced once, politely. Screen readers get "Loading …" from the
 *   live region; the bars themselves are decorative and hidden, rather than
 *   being read out as a dozen empty list items.
 */

import clsx from 'clsx';
import React from 'react';

interface SkeletonListProps {
  /** How many placeholder rows to draw. Match the page's usual first screen. */
  rows?: number;
  /** `card` for a list of bordered cards, `row` for a dense table-like list. */
  variant?: 'card' | 'row';
  /** What is loading, for the announcement. */
  label?: string;
  className?: string;
}

/** One shimmering bar. Width varies so the block does not read as a grid.
 *  The delay has to sit on the animated element itself: `animation-delay` is
 *  not inherited, so setting it on the row would have done nothing. */
const Bar: React.FC<{ className?: string; delay: number }> = ({ className, delay }) => (
  <div className={clsx('skeleton h-3', className)} style={{ animationDelay: `${delay}ms` }} />
);

const SkeletonList: React.FC<SkeletonListProps> = ({
  rows = 5,
  variant = 'card',
  label = 'Loading',
  className,
}) => (
  <div
    className={clsx(variant === 'card' ? 'space-y-3' : 'space-y-2', className)}
    role="status"
    aria-live="polite"
    aria-busy="true"
  >
    <span className="sr-only">{label}</span>

    {Array.from({ length: rows }).map((_, i) => (
      <div
        key={i}
        aria-hidden="true"
        className={clsx(
          variant === 'card'
            ? 'bg-white border border-gray-200 rounded-lg p-4'
            : 'px-3 py-2 border-b border-gray-200'
        )}
      >
        <div className="flex items-start gap-3">
          {/* Each row's shimmer starts slightly after the one above, so the
              sweep travels down the list instead of flashing all at once. */}
          {variant === 'card' && (
            <div
              className="skeleton w-8 h-8 rounded-md shrink-0"
              style={{ animationDelay: `${i * 90}ms` }}
            />
          )}
          <div className="flex-1 min-w-0 space-y-2">
            {/* Title, then a shorter line of metadata: the shape almost every
                card on this app's list pages actually has. */}
            <Bar
              delay={i * 90}
              className={i % 3 === 0 ? 'w-2/3' : i % 3 === 1 ? 'w-1/2' : 'w-3/5'}
            />
            <Bar delay={i * 90} className={clsx('h-2', i % 2 === 0 ? 'w-1/3' : 'w-2/5')} />
          </div>
          {variant === 'card' && (
            <div
              className="skeleton w-16 h-5 rounded-full shrink-0"
              style={{ animationDelay: `${i * 90}ms` }}
            />
          )}
        </div>
      </div>
    ))}
  </div>
);

export default SkeletonList;
