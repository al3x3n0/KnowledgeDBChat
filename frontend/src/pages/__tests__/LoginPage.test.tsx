/**
 * Tests for LoginPage component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import LoginPage from '../LoginPage';
import { AuthProvider } from '../../contexts/AuthContext';

// Mock the auth context
jest.mock('../../contexts/AuthContext', () => ({
  ...jest.requireActual('../../contexts/AuthContext'),
  useAuth: () => ({
    login: jest.fn().mockResolvedValue({}),
    register: jest.fn().mockResolvedValue({}),
    user: null,
    loading: false,
  }),
}));

jest.mock('../../services/api', () => ({
  ...jest.requireActual('../../services/api'),
  apiClient: {
    ...jest.requireActual('../../services/api').apiClient,
    getSignInOptions: jest.fn().mockResolvedValue(null),
  },
}));

// eslint-disable-next-line import/first
import { apiClient } from '../../services/api';

const options = apiClient.getSignInOptions as jest.Mock;

const renderWithRouter = (component: React.ReactElement) => {
  return render(
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AuthProvider>
        {component}
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('LoginPage', () => {
  beforeEach(() => {
    // Mocks are reset before each test (CRA's resetMocks), which drops a
    // default given where the mock is made.
    options.mockResolvedValue(null);
  });

  it('renders login form by default', () => {
    renderWithRouter(<LoginPage />);
    expect(screen.getByText(/sign in to your account/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('switches to register form', () => {
    renderWithRouter(<LoginPage />);
    const switchButton = screen.getByText(/have an account\?\s*sign up/i);
    fireEvent.click(switchButton);
    
    expect(screen.getByText(/create your account/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
  });

  it('validates required fields in login form', async () => {
    renderWithRouter(<LoginPage />);
    const submitButton = screen.getByRole('button', { name: /sign in/i });
    fireEvent.click(submitButton);
    
    // Form validation should prevent submission
    await waitFor(() => {
      expect(screen.getByText(/username is required/i)).toBeInTheDocument();
    });
  });

  it('validates password match in register form', async () => {
    renderWithRouter(<LoginPage />);
    
    // Switch to register
    const switchButton = screen.getByText(/have an account\?\s*sign up/i);
    fireEvent.click(switchButton);

    fireEvent.change(screen.getByLabelText(/^username$/i), { target: { value: 'testuser' } });
    fireEvent.change(screen.getByLabelText(/^email$/i), { target: { value: 'test@example.com' } });

    const passwordInput = screen.getByLabelText(/^password$/i);
    const confirmPasswordInput = screen.getByLabelText(/confirm password/i);
    
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    fireEvent.change(confirmPasswordInput, { target: { value: 'different' } });
    
    const submitButton = screen.getByRole('button', { name: /create account/i });
    fireEvent.click(submitButton);
    
    // Should show validation error
    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  it('offers no sign-up where accounts come from the directory', async () => {
    options.mockResolvedValueOnce({ self_registration: false, directory: true });
    renderWithRouter(<LoginPage />);

    expect(await screen.findByText(/sign in with your directory account/i)).toBeInTheDocument();
    expect(screen.queryByText(/sign up/i)).not.toBeInTheDocument();
  });

  it('keeps sign-up when the options cannot be read', async () => {
    options.mockResolvedValueOnce(null);
    renderWithRouter(<LoginPage />);

    await waitFor(() => expect(options).toHaveBeenCalled());
    expect(screen.getByText(/have an account\?\s*sign up/i)).toBeInTheDocument();
  });
});
