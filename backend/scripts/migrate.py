#!/usr/bin/env python3
"""
Database migration management script for Knowledge Database
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.database import create_tables, drop_tables
from app.core.config import settings

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

async def create_initial_migration():
    """Create the initial migration"""
    print("📝 Creating initial migration...")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Create initial migration
    command = "alembic revision --autogenerate -m 'Initial migration'"
    return run_command(command, "Creating initial migration")

async def run_migrations():
    """Run pending migrations"""
    print("🚀 Running database migrations...")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Run migrations
    command = "alembic upgrade head"
    return run_command(command, "Running migrations")

async def rollback_migration():
    """Rollback the last migration"""
    print("⏪ Rolling back last migration...")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Rollback one migration
    command = "alembic downgrade -1"
    return run_command(command, "Rolling back migration")

async def show_migration_history():
    """Show migration history"""
    print("📋 Migration history:")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Show history
    command = "alembic history"
    return run_command(command, "Showing migration history")

async def show_current_revision():
    """Show current database revision"""
    print("📍 Current database revision:")
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Show current revision
    command = "alembic current"
    return run_command(command, "Showing current revision")

async def reset_database():
    """Reset the database (drop and recreate all tables)"""
    print("⚠️  WARNING: This will delete all data in the database!")
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Database reset cancelled.")
        return False
    
    print("🗑️  Resetting database...")
    
    try:
        # Drop all tables
        await drop_tables()
        print("✅ All tables dropped")
        
        # Create all tables
        await create_tables()
        print("✅ All tables created")
        
        print("✅ Database reset completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False

async def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "init":
            await create_initial_migration()
        elif command == "upgrade":
            await run_migrations()
        elif command == "downgrade":
            await rollback_migration()
        elif command == "history":
            await show_migration_history()
        elif command == "current":
            await show_current_revision()
        elif command == "reset":
            await reset_database()
        else:
            print("❌ Unknown command. Available commands:")
            print("  init      - Create initial migration")
            print("  upgrade   - Run pending migrations")
            print("  downgrade - Rollback last migration")
            print("  history   - Show migration history")
            print("  current   - Show current revision")
            print("  reset     - Reset database (WARNING: deletes all data)")
            sys.exit(1)
    else:
        print("🗄️  Knowledge Database - Migration Management")
        print("=" * 50)
        print("Available commands:")
        print("  python migrate.py init      - Create initial migration")
        print("  python migrate.py upgrade   - Run pending migrations")
        print("  python migrate.py downgrade - Rollback last migration")
        print("  python migrate.py history   - Show migration history")
        print("  python migrate.py current   - Show current revision")
        print("  python migrate.py reset     - Reset database (WARNING: deletes all data)")
        print()
        print("Or run without arguments for interactive mode:")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Create initial migration")
            print("2. Run migrations")
            print("3. Rollback last migration")
            print("4. Show migration history")
            print("5. Show current revision")
            print("6. Reset database (WARNING: deletes all data)")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                await create_initial_migration()
            elif choice == "2":
                await run_migrations()
            elif choice == "3":
                await rollback_migration()
            elif choice == "4":
                await show_migration_history()
            elif choice == "5":
                await show_current_revision()
            elif choice == "6":
                await reset_database()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    asyncio.run(main())








