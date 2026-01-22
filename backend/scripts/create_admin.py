#!/usr/bin/env python3
"""
Script to create an admin user for the Knowledge Database system
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
from typing import Optional

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import AuthService
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import getpass

def _get_password_from_env(var_name: Optional[str]) -> Optional[str]:
    if not var_name:
        return None
    return os.getenv(var_name)


async def reset_user_password(
    identifier: Optional[str] = None,
    password: Optional[str] = None,
    password_env: Optional[str] = None,
):
    """Reset a user's password (admin recovery)."""
    print("🔑 Knowledge Database - Password Reset")
    print("=" * 50)

    async for db in get_db():
        auth_service = AuthService()

        if password is None:
            password = _get_password_from_env(password_env)

        # Pick a user
        user = None
        while user is None:
            if not identifier:
                identifier = input("Username or Email to reset: ").strip()
            if not identifier:
                print("❌ Identifier cannot be empty.")
                identifier = None
                continue

            result = await db.execute(
                select(User).where((User.username == identifier) | (User.email == identifier))
            )
            user = result.scalar_one_or_none()
            if not user:
                print(f"❌ No user found for '{identifier}'. Try again.")
                identifier = None

        print(f"Found user: {user.username} ({user.email}) role={user.role} active={user.is_active}")

        # Get new password
        if password is None:
            if not sys.stdin.isatty():
                raise SystemExit(
                    "Password required. Use interactive TTY, or pass --password, or --password-env VAR."
                )
            while True:
                password = getpass.getpass("New Password: ")
                if not password:
                    print("❌ Password cannot be empty.")
                    continue
                if len(password) < 8:
                    print("❌ Password must be at least 8 characters long.")
                    continue

                confirm_password = getpass.getpass("Confirm New Password: ")
                if password != confirm_password:
                    print("❌ Passwords do not match.")
                    continue
                break
        else:
            if len(password) < 8:
                raise SystemExit("Password must be at least 8 characters long.")

        try:
            print("\n🔄 Updating password...")
            user.hashed_password = auth_service.get_password_hash(password)
            await db.commit()
            await db.refresh(user)
            print("✅ Password updated successfully.")
        except Exception as e:
            print(f"❌ Error updating password: {e}")
            await db.rollback()

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

async def create_admin_user_noninteractive(
    username: str,
    email: str,
    password: str,
    full_name: Optional[str] = None,
):
    """Create an admin user without prompts (for CI/bootstrap)."""
    print("🔧 Knowledge Database - Admin User Creation (non-interactive)")
    print("=" * 50)

    async for db in get_db():
        auth_service = AuthService()

        # Validate uniqueness
        result = await db.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none():
            raise SystemExit(f"Username '{username}' already exists.")

        result = await db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            raise SystemExit(f"Email '{email}' already exists.")

        if len(password) < 8:
            raise SystemExit("Password must be at least 8 characters long.")

        try:
            print("🔄 Creating admin user...")
            hashed_password = auth_service.get_password_hash(password)
            admin_user = User(
                username=username,
                email=email,
                full_name=full_name if full_name else None,
                hashed_password=hashed_password,
                role="admin",
                is_active=True,
                is_verified=True,
            )
            db.add(admin_user)
            await db.commit()
            await db.refresh(admin_user)
            print("✅ Admin user created successfully!")
            print(f"   Username: {admin_user.username}")
            print(f"   Email: {admin_user.email}")
            print(f"   Role: {admin_user.role}")
            print(f"   User ID: {admin_user.id}")
        except Exception as e:
            print(f"❌ Error creating admin user: {e}")
            await db.rollback()

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
    parser = argparse.ArgumentParser(description="Knowledge Database user management")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all users")
    p_admin = sub.add_parser("admin", help="Create an admin user")
    p_admin.add_argument("--username", help="Username (non-interactive)")
    p_admin.add_argument("--email", help="Email (non-interactive)")
    p_admin.add_argument("--full-name", help="Full name (optional)")
    p_admin.add_argument("--password", help="Password (avoid using in shell history)")
    p_admin.add_argument("--password-env", help="Env var name to read password from")
    sub.add_parser("user", help="Create a regular user (interactive)")

    p_reset = sub.add_parser("reset", help="Reset a user's password")
    p_reset.add_argument("identifier", nargs="?", help="Username or email")
    p_reset.add_argument("--password", help="New password (avoid using in shell history)")
    p_reset.add_argument(
        "--password-env",
        help="Env var name to read new password from (recommended for non-interactive use)",
    )

    args, _ = parser.parse_known_args()

    if args.command:
        if args.command == "admin":
            password = args.password or _get_password_from_env(args.password_env)
            if args.username and args.email and password:
                await create_admin_user_noninteractive(
                    username=args.username,
                    email=args.email,
                    password=password,
                    full_name=args.full_name,
                )
            else:
                await create_admin_user()
        elif args.command == "user":
            await create_regular_user()
        elif args.command == "list":
            await list_users()
        elif args.command == "reset":
            await reset_user_password(
                identifier=args.identifier,
                password=args.password,
                password_env=args.password_env,
            )
        else:
            raise SystemExit("Unknown command")
    else:
        print("🔧 Knowledge Database - User Management")
        print("=" * 50)
        print("Available commands:")
        print("  python create_admin.py admin  - Create an admin user")
        print("  python create_admin.py user   - Create a regular user")
        print("  python create_admin.py list   - List all users")
        print("  python create_admin.py reset [username_or_email] - Reset a user's password")
        print()
        print("Or run without arguments for interactive mode:")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Create admin user")
            print("2. Create regular user")
            print("3. List all users")
            print("4. Reset user password")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                await create_admin_user()
            elif choice == "2":
                await create_regular_user()
            elif choice == "3":
                await list_users()
            elif choice == "4":
                await reset_user_password()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    asyncio.run(main())






