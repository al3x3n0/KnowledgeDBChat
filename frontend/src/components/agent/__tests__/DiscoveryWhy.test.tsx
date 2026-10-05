import React from 'react';
import { render, screen } from '@testing-library/react';

import { DiscoveryWhy, readDiscoverySignals } from '../DiscoveryWhy';

const favourable = {
  kind: 'phrase' as const,
  term: 'sparse attention',
  weight: 3,
  favourable: true,
  label: '“sparse attention”',
  text: 'matches “sparse attention”, a phrase from items you kept',
};

const against = {
  kind: 'token' as const,
  term: 'blockchain',
  weight: -6,
  favourable: false,
  label: '“blockchain”',
  text: 'matches “blockchain”, a word from items you dismissed',
};

describe('DiscoveryWhy', () => {
  it('names the term that surfaced the item', () => {
    render(<DiscoveryWhy metadata={{ discovery_signals: [favourable] }} />);
    expect(screen.getByText(/Surfaced because it matches/)).toBeInTheDocument();
    expect(screen.getByText('“sparse attention”')).toBeInTheDocument();
  });

  it('does not say an item was surfaced because of a term you dismissed', () => {
    // The contradiction this component exists to avoid: a negative signal
    // explains a ranking down, not a surfacing.
    render(<DiscoveryWhy metadata={{ discovery_signals: [against] }} />);
    expect(screen.queryByText(/Surfaced because/)).not.toBeInTheDocument();
    expect(screen.getByText(/Ranked down by/)).toBeInTheDocument();
  });

  it('joins the two with "despite" when both are present', () => {
    render(<DiscoveryWhy metadata={{ discovery_signals: [favourable, against] }} />);
    expect(screen.getByText(/Surfaced because it matches/)).toBeInTheDocument();
    expect(screen.getByText('despite')).toBeInTheDocument();
  });

  it('keeps the exact weight available without claiming it is a count', () => {
    render(<DiscoveryWhy metadata={{ discovery_signals: [favourable] }} />);
    const chip = screen.getByText('“sparse attention”');
    expect(chip.getAttribute('title')).toContain('learned weight +3');
    expect(chip.textContent).not.toContain('3');
  });

  it('falls back to the reasons rows recorded before signals existed', () => {
    render(<DiscoveryWhy metadata={{ discovery_reasons: ['token_bias'] }} />);
    expect(screen.getByText(/Why this: token_bias/)).toBeInTheDocument();
  });

  it('renders nothing when the profile had no opinion', () => {
    const { container } = render(<DiscoveryWhy metadata={{}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('survives a malformed signal rather than crashing the inbox', () => {
    // TypeScript types are erased at runtime; the server is the only guarantee
    // and it is deployed separately.
    expect(readDiscoverySignals({ discovery_signals: [null, 'x', { label: 'ok' }] })).toEqual([]);
    const { container } = render(
      <DiscoveryWhy metadata={{ discovery_signals: 'not an array' }} />
    );
    expect(container).toBeEmptyDOMElement();
  });
});
