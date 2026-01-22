/**
 * Login page component
 */

import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { Database, Mail, Lock, User } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import Input from '../components/common/Input';

interface LoginForm {
  username: string;
  password: string;
}

interface RegisterForm {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  full_name?: string;
}

const LoginPage: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const loginForm = useForm<LoginForm>();
  const registerForm = useForm<RegisterForm>();

  const onLoginSubmit = async (data: LoginForm) => {
    try {
      setLoading(true);
      await login(data.username, data.password);
      navigate('/chat');
    } catch (error) {
      // Error is handled in the auth context
    } finally {
      setLoading(false);
    }
  };

  const onRegisterSubmit = async (data: RegisterForm) => {
    if (data.password !== data.confirmPassword) {
      registerForm.setError('confirmPassword', {
        type: 'manual',
        message: 'Passwords do not match'
      });
      return;
    }

    try {
      setLoading(true);
      await register({
        username: data.username,
        email: data.email,
        password: data.password,
        full_name: data.full_name,
      });
      navigate('/chat');
    } catch (error) {
      // Error is handled in the auth context
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <div className="flex justify-center">
            <div className="bg-primary-600 p-3 rounded-full">
              <Database className="w-8 h-8 text-white" />
            </div>
          </div>
          <h2 className="mt-6 text-3xl font-bold text-gray-900">
            Knowledge Database
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </p>
        </div>

        {/* Form */}
        <div className="bg-white py-8 px-6 shadow-xl rounded-lg">
          {isLogin ? (
            <form onSubmit={loginForm.handleSubmit(onLoginSubmit)} className="space-y-6">
              <Input
                label="Username"
                type="text"
                fullWidth
                leftIcon={<User className="w-4 h-4" />}
                error={loginForm.formState.errors.username?.message}
                {...loginForm.register('username', {
                  required: 'Username is required',
                  minLength: {
                    value: 3,
                    message: 'Username must be at least 3 characters'
                  }
                })}
              />

              <Input
                label="Password"
                type="password"
                fullWidth
                leftIcon={<Lock className="w-4 h-4" />}
                error={loginForm.formState.errors.password?.message}
                {...loginForm.register('password', {
                  required: 'Password is required'
                })}
              />

              <Button
                type="submit"
                fullWidth
                loading={loading}
                disabled={loading}
              >
                Sign In
              </Button>
            </form>
          ) : (
            <form onSubmit={registerForm.handleSubmit(onRegisterSubmit)} className="space-y-6">
              <Input
                label="Username"
                type="text"
                fullWidth
                leftIcon={<User className="w-4 h-4" />}
                error={registerForm.formState.errors.username?.message}
                {...registerForm.register('username', {
                  required: 'Username is required',
                  minLength: {
                    value: 3,
                    message: 'Username must be at least 3 characters'
                  }
                })}
              />

              <Input
                label="Email"
                type="email"
                fullWidth
                leftIcon={<Mail className="w-4 h-4" />}
                error={registerForm.formState.errors.email?.message}
                {...registerForm.register('email', {
                  required: 'Email is required',
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: 'Invalid email address'
                  }
                })}
              />

              <Input
                label="Full Name (Optional)"
                type="text"
                fullWidth
                {...registerForm.register('full_name')}
              />

              <Input
                label="Password"
                type="password"
                fullWidth
                leftIcon={<Lock className="w-4 h-4" />}
                error={registerForm.formState.errors.password?.message}
                {...registerForm.register('password', {
                  required: 'Password is required',
                  minLength: {
                    value: 6,
                    message: 'Password must be at least 6 characters'
                  }
                })}
              />

              <Input
                label="Confirm Password"
                type="password"
                fullWidth
                leftIcon={<Lock className="w-4 h-4" />}
                error={registerForm.formState.errors.confirmPassword?.message}
                {...registerForm.register('confirmPassword', {
                  required: 'Please confirm your password'
                })}
              />

              <Button
                type="submit"
                fullWidth
                loading={loading}
                disabled={loading}
              >
                Create Account
              </Button>
            </form>
          )}

          {/* Toggle between login and register */}
          <div className="mt-6 text-center">
            <button
              type="button"
              className="text-sm text-primary-600 hover:text-primary-500"
              onClick={() => {
                setIsLogin(!isLogin);
                loginForm.reset();
                registerForm.reset();
              }}
            >
              {isLogin 
                ? "Don't have an account? Sign up" 
                : "Already have an account? Sign in"
              }
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-xs text-gray-500">
          <p>
            Secure organizational knowledge management with AI-powered search
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;









