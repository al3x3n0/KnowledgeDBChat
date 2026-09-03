/**
 * Tests for Button component
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Button from '../Button';

describe('Button', () => {
  it('renders button with text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('calls onClick handler when clicked', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    
    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('renders disabled button', () => {
    render(<Button disabled>Disabled</Button>);
    const button = screen.getByText('Disabled');
    expect(button).toBeDisabled();
  });

  it('does not call onClick when disabled', () => {
    const handleClick = jest.fn();
    render(<Button disabled onClick={handleClick}>Disabled</Button>);
    
    fireEvent.click(screen.getByText('Disabled'));
    expect(handleClick).not.toHaveBeenCalled();
  });

  // These assert the distinctions the variants exist to make, not the exact
  // utilities that express them: only `primary` carries the accent, only
  // `danger` carries red, and `ghost` has no border at rest. Asserting the
  // full utility string made restyling a test failure rather than a review.
  it('gives only the primary variant the accent', () => {
    const { rerender } = render(<Button variant="primary">Primary</Button>);
    expect(screen.getByText('Primary').className).toContain('primary-');

    rerender(<Button variant="secondary">Secondary</Button>);
    expect(screen.getByText('Secondary').className).not.toContain('primary-');
  });

  it('marks the destructive variant in red, and no other', () => {
    const { rerender } = render(<Button variant="danger">Delete</Button>);
    expect(screen.getByText('Delete').className).toContain('red-');

    rerender(<Button variant="ghost">Ghost</Button>);
    expect(screen.getByText('Ghost').className).not.toContain('red-');
    // A ghost button should be invisible until wanted.
    expect(screen.getByText('Ghost')).toHaveClass('border-transparent');
  });

  it('shows a spinner and blocks the click while loading', () => {
    const handleClick = jest.fn();
    render(
      <Button loading onClick={handleClick}>
        Saving
      </Button>
    );
    const button = screen.getByText('Saving');
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('aria-busy', 'true');

    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });
});
