/**
 * Authentication context for managing user state
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import { AxiosError } from 'axios';
import { apiClient, LoginResponse } from '../services/api';
import { User } from '../types';
import toast from 'react-hot-toast';

// Error response structure from API
interface ApiErrorResponse {
  detail?: string;
  message?: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (userData: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (user: User) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Check if user is already logged in on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('access_token');
        if (token) {
          const currentUser = await apiClient.getCurrentUser();
          setUser(currentUser);
        }
      } catch (error) {
        // Token is invalid, clear it
        localStorage.removeItem('access_token');
        apiClient.clearToken();
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  const login = async (username: string, password: string): Promise<void> => {
    try {
      setLoading(true);
      const response: LoginResponse = await apiClient.login(username, password);

      // Validate response structure
      if (!response || !response.access_token) {
        throw new Error('Invalid login response: missing access token');
      }

      if (!response.user) {
        throw new Error('Invalid login response: missing user data');
      }

      // Store token and user data
      apiClient.setToken(response.access_token);
      setUser(response.user);
      toast.success(`Welcome back, ${response.user.username}!`);
    } catch (error) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      const errorMessage = axiosError.response?.data?.detail || axiosError.message || 'Login failed';
      toast.error(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const register = async (userData: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
  }): Promise<void> => {
    try {
      setLoading(true);
      const newUser = await apiClient.register(userData);

      // After registration, automatically log in
      await login(userData.username, userData.password);

      toast.success(`Welcome to Knowledge Database, ${newUser.username}!`);
    } catch (error) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      const errorMessage = axiosError.response?.data?.detail || 'Registration failed';
      toast.error(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const logout = async (): Promise<void> => {
    try {
      await apiClient.logout();
    } catch (error) {
      // Even if logout API fails, clear local state
      console.error('Logout API error:', error);
    } finally {
      // Clear token and user data
      apiClient.clearToken();
      setUser(null);
      toast.success('Logged out successfully');
    }
  };

  const updateUser = (updatedUser: User) => {
    setUser(updatedUser);
  };

  const value: AuthContextType = {
    user,
    loading,
    login,
    register,
    logout,
    updateUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};


