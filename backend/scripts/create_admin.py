#!/usr/bin/env python3
"""
Script to create an admin user for the Knowledge Database system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import AuthService
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import getpass

async def create_admin_user():
    """Create an admin user interactively"""
    print("🔧 Knowledge Database - Admin User Creation")
    print("=" * 50)
    
    # Get database connection
    async for db in get_db():
        auth_service = AuthService()
        
        # Check if admin users already exist
        result = await db.execute(select(User).where(User.role == "admin"))
        existing_admins = result.scalars().all()
        
        if existing_admins:
            print(f"⚠️  Found {len(existing_admins)} existing admin user(s):")
            for admin in existing_admins:
                print(f"   - {admin.username} ({admin.email})")
            
            response = input("\nDo you want to create another admin user? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Admin user creation cancelled.")
                return
        
        print("\n📝 Please provide the following information:")
        
        # Get user input
        while True:
            username = input("Username: ").strip()
            if not username:
                print("❌ Username cannot be empty.")
                continue
            if len(username) < 3:
                print("❌ Username must be at least 3 characters long.")
                continue
            
            # Check if username already exists
            result = await db.execute(select(User).where(User.username == username))
            existing_user = result.scalar_one_or_none()
            if existing_user:
                print(f"❌ Username '{username}' already exists.")
                continue
            break
        
        while True:
            email = input("Email: ").strip()
            if not email:
                print("❌ Email cannot be empty.")
                continue
            if "@" not in email:
                print("❌ Please enter a valid email address.")
                continue
            
            # Check if email already exists
            result = await db.execute(select(User).where(User.email == email))
            existing_user = result.scalar_one_or_none()
            if existing_user:
                print(f"❌ Email '{email}' already exists.")
                continue
            break
        
        full_name = input("Full Name (optional): ").strip()
        
        while True:
            password = getpass.getpass("Password: ")
            if not password:
                print("❌ Password cannot be empty.")
                continue
            if len(password) < 6:
                print("❌ Password must be at least 6 characters long.")
                continue
            
            confirm_password = getpass.getpass("Confirm Password: ")
            if password != confirm_password:
                print("❌ Passwords do not match.")
                continue
            break
        
        # Create the admin user
        try:
            print("\n🔄 Creating admin user...")
            
            # Hash the password
            hashed_password = auth_service.get_password_hash(password)
            
            # Create user object
            admin_user = User(
                username=username,
                email=email,
                full_name=full_name if full_name else None,
                hashed_password=hashed_password,
                role="admin",
                is_active=True,
                is_verified=True
            )
            
            # Add to database
            db.add(admin_user)
            await db.commit()
            await db.refresh(admin_user)
            
            print("✅ Admin user created successfully!")
            print(f"   Username: {admin_user.username}")
            print(f"   Email: {admin_user.email}")
            print(f"   Role: {admin_user.role}")
            print(f"   User ID: {admin_user.id}")
            
            print("\n🚀 You can now login to the admin dashboard at:")
            print("   http://localhost:3000/admin")
            
        except Exception as e:
            print(f"❌ Error creating admin user: {e}")
            await db.rollback()
            return
        
        break

async def create_regular_user():
    """Create a regular user interactively"""
    print("👤 Knowledge Database - User Registration")
    print("=" * 50)
    
    # Get database connection
    async for db in get_db():
        auth_service = AuthService()
        
        print("\n📝 Please provide the following information:")
        
        # Get user input
        while True:
            username = input("Username: ").strip()
            if not username:
                print("❌ Username cannot be empty.")
                continue
            if len(username) < 3:
                print("❌ Username must be at least 3 characters long.")
                continue
            
            # Check if username already exists
            result = await db.execute(select(User).where(User.username == username))
            existing_user = result.scalar_one_or_none()
            if existing_user:
                print(f"❌ Username '{username}' already exists.")
                continue
            break
        
        while True:
            email = input("Email: ").strip()
            if not email:
                print("❌ Email cannot be empty.")
                continue
            if "@" not in email:
                print("❌ Please enter a valid email address.")
                continue
            
            # Check if email already exists
            result = await db.execute(select(User).where(User.email == email))
            existing_user = result.scalar_one_or_none()
            if existing_user:
                print(f"❌ Email '{email}' already exists.")
                continue
            break
        
        full_name = input("Full Name (optional): ").strip()
        
        while True:
            password = getpass.getpass("Password: ")
            if not password:
                print("❌ Password cannot be empty.")
                continue
            if len(password) < 6:
                print("❌ Password must be at least 6 characters long.")
                continue
            
            confirm_password = getpass.getpass("Confirm Password: ")
            if password != confirm_password:
                print("❌ Passwords do not match.")
                continue
            break
        
        # Create the user
        try:
            print("\n🔄 Creating user...")
            
            # Hash the password
            hashed_password = auth_service.get_password_hash(password)
            
            # Create user object
            user = User(
                username=username,
                email=email,
                full_name=full_name if full_name else None,
                hashed_password=hashed_password,
                role="user",
                is_active=True,
                is_verified=False  # Regular users need verification
            )
            
            # Add to database
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            print("✅ User created successfully!")
            print(f"   Username: {user.username}")
            print(f"   Email: {user.email}")
            print(f"   Role: {user.role}")
            print(f"   User ID: {user.id}")
            print(f"   Status: {'Active' if user.is_active else 'Inactive'}")
            print(f"   Verified: {'Yes' if user.is_verified else 'No'}")
            
            print("\n🚀 You can now login at:")
            print("   http://localhost:3000/login")
            
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            await db.rollback()
            return
        
        break

async def list_users():
    """List all users in the system"""
    print("👥 Knowledge Database - User List")
    print("=" * 50)
    
    async for db in get_db():
        result = await db.execute(select(User).order_by(User.created_at.desc()))
        users = result.scalars().all()
        
        if not users:
            print("❌ No users found in the system.")
            return
        
        print(f"\nFound {len(users)} user(s):")
        print("-" * 80)
        print(f"{'Username':<20} {'Email':<30} {'Role':<10} {'Status':<10}")
        print("-" * 80)
        
        for user in users:
            status = "Active" if user.is_active else "Inactive"
            verified = "✓" if user.is_verified else "✗"
            print(f"{user.username:<20} {user.email:<30} {user.role:<10} {status} {verified}")
        
        print("-" * 80)

async def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "admin":
            await create_admin_user()
        elif command == "user":
            await create_regular_user()
        elif command == "list":
            await list_users()
        else:
            print("❌ Unknown command. Use: admin, user, or list")
            sys.exit(1)
    else:
        print("🔧 Knowledge Database - User Management")
        print("=" * 50)
        print("Available commands:")
        print("  python create_admin.py admin  - Create an admin user")
        print("  python create_admin.py user   - Create a regular user")
        print("  python create_admin.py list   - List all users")
        print()
        print("Or run without arguments for interactive mode:")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Create admin user")
            print("2. Create regular user")
            print("3. List all users")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                await create_admin_user()
            elif choice == "2":
                await create_regular_user()
            elif choice == "3":
                await list_users()
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    asyncio.run(main())







