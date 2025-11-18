/**
 * Settings page for user preferences and account management
 */

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { useMutation } from 'react-query';
import { User, Lock, Bell, Palette, Shield } from 'lucide-react';

import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../services/api';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import toast from 'react-hot-toast';

interface PasswordChangeForm {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

const SettingsPage: React.FC = () => {
  const { user, updateUser } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');

  const tabs = [
    { id: 'profile', name: 'Profile', icon: User },
    { id: 'security', name: 'Security', icon: Lock },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'appearance', name: 'Appearance', icon: Palette },
  ];

  // Add admin tab if user is admin
  if (user?.role === 'admin') {
    tabs.push({ id: 'admin', name: 'Administration', icon: Shield });
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Manage your account preferences and settings</p>
      </div>

      <div className="flex space-x-8">
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200 ${
                    activeTab === tab.id
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  {tab.name}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1">
          {activeTab === 'profile' && <ProfileTab />}
          {activeTab === 'security' && <SecurityTab />}
          {activeTab === 'notifications' && <NotificationsTab />}
          {activeTab === 'appearance' && <AppearanceTab />}
          {activeTab === 'admin' && user?.role === 'admin' && <AdminTab />}
        </div>
      </div>
    </div>
  );
};

// Profile Tab
const ProfileTab: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Profile Information</h2>
      
      <div className="space-y-6">
        {/* Avatar */}
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center">
            <User className="w-8 h-8 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900">Profile Picture</h3>
            <p className="text-sm text-gray-500">Upload a profile picture to personalize your account</p>
            <Button variant="ghost" size="sm" className="mt-2">
              Change Picture
            </Button>
          </div>
        </div>

        {/* User Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Input
            label="Username"
            value={user?.username || ''}
            disabled
            helpText="Username cannot be changed"
          />
          
          <Input
            label="Email"
            value={user?.email || ''}
            disabled
            helpText="Contact admin to change email"
          />
          
          <Input
            label="Full Name"
            value={user?.full_name || ''}
            placeholder="Enter your full name"
          />
          
          <Input
            label="Role"
            value={user?.role || ''}
            disabled
            helpText="Role is assigned by administrators"
          />
        </div>

        {/* Account Status */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Account Status</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Account Status:</span>
              <span className={`font-medium ${user?.is_active ? 'text-green-600' : 'text-red-600'}`}>
                {user?.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Verification Status:</span>
              <span className={`font-medium ${user?.is_verified ? 'text-green-600' : 'text-yellow-600'}`}>
                {user?.is_verified ? 'Verified' : 'Pending'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Member Since:</span>
              <span className="text-gray-600">
                {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <Button>Save Changes</Button>
        </div>
      </div>
    </div>
  );
};

// Security Tab
const SecurityTab: React.FC = () => {
  const { user } = useAuth();
  const { register, handleSubmit, formState: { errors }, watch, reset } = useForm<PasswordChangeForm>();

  const changePasswordMutation = useMutation(
    async (data: { current_password: string; new_password: string }) => {
      // This endpoint doesn't exist yet, but the structure is ready
      const response = await fetch('/api/v1/users/me/password', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        throw new Error('Failed to change password');
      }
      
      return response.json();
    },
    {
      onSuccess: () => {
        toast.success('Password changed successfully');
        reset();
      },
      onError: () => {
        toast.error('Failed to change password');
      },
    }
  );

  const onSubmit = (data: PasswordChangeForm) => {
    if (data.newPassword !== data.confirmPassword) {
      toast.error('Passwords do not match');
      return;
    }

    changePasswordMutation.mutate({
      current_password: data.currentPassword,
      new_password: data.newPassword,
    });
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Security Settings</h2>
      
      <div className="space-y-8">
        {/* Password Change */}
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Change Password</h3>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 max-w-md">
            <Input
              label="Current Password"
              type="password"
              error={errors.currentPassword?.message}
              {...register('currentPassword', {
                required: 'Current password is required'
              })}
            />
            
            <Input
              label="New Password"
              type="password"
              error={errors.newPassword?.message}
              {...register('newPassword', {
                required: 'New password is required',
                minLength: {
                  value: 6,
                  message: 'Password must be at least 6 characters'
                }
              })}
            />
            
            <Input
              label="Confirm New Password"
              type="password"
              error={errors.confirmPassword?.message}
              {...register('confirmPassword', {
                required: 'Please confirm your password'
              })}
            />
            
            <Button
              type="submit"
              loading={changePasswordMutation.isLoading}
            >
              Change Password
            </Button>
          </form>
        </div>

        {/* Login History */}
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Account Activity</h3>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Last Login:</span>
                <span className="text-gray-600">
                  {user?.last_login ? new Date(user.last_login).toLocaleString() : 'Never'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Total Logins:</span>
                <span className="text-gray-600">{user?.login_count || 0}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Notifications Tab
const NotificationsTab: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Notification Preferences</h2>
      
      <div className="space-y-6">
        <div className="space-y-4">
          <label className="flex items-start space-x-3">
            <input type="checkbox" className="mt-1" defaultChecked />
            <div>
              <div className="font-medium text-gray-900">Email Notifications</div>
              <div className="text-sm text-gray-500">Receive email notifications for important updates</div>
            </div>
          </label>
          
          <label className="flex items-start space-x-3">
            <input type="checkbox" className="mt-1" defaultChecked />
            <div>
              <div className="font-medium text-gray-900">Document Processing</div>
              <div className="text-sm text-gray-500">Notify when document processing is complete</div>
            </div>
          </label>
          
          <label className="flex items-start space-x-3">
            <input type="checkbox" className="mt-1" />
            <div>
              <div className="font-medium text-gray-900">System Maintenance</div>
              <div className="text-sm text-gray-500">Receive notifications about system maintenance</div>
            </div>
          </label>
        </div>
        
        <div className="flex justify-end">
          <Button>Save Preferences</Button>
        </div>
      </div>
    </div>
  );
};

// Appearance Tab
const AppearanceTab: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Appearance Settings</h2>
      
      <div className="space-y-6">
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Theme</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-3">
              <input type="radio" name="theme" value="light" defaultChecked />
              <span>Light Theme</span>
            </label>
            <label className="flex items-center space-x-3">
              <input type="radio" name="theme" value="dark" />
              <span>Dark Theme</span>
            </label>
            <label className="flex items-center space-x-3">
              <input type="radio" name="theme" value="auto" />
              <span>Auto (System Preference)</span>
            </label>
          </div>
        </div>
        
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Language</h3>
          <select className="block w-48 rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500">
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
          </select>
        </div>
        
        <div className="flex justify-end">
          <Button>Save Settings</Button>
        </div>
      </div>
    </div>
  );
};

// Admin Tab
const AdminTab: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Administration</h2>
      
      <div className="space-y-6">
        <p className="text-gray-600">
          Administrative functions are available in the dedicated Admin dashboard.
        </p>
        
        <Button onClick={() => window.open('/admin', '_blank')}>
          Open Admin Dashboard
        </Button>
      </div>
    </div>
  );
};

export default SettingsPage;







