#!/usr/bin/env python3
"""
Setup script for VoiceForm Data API configuration.

This script helps you set up your .env file for the data API client.

Run with: python -m conversion_subnet.data_api.setup_env
"""

import shutil
from pathlib import Path

def main():
    """Setup .env configuration file."""
    
    print("🔧 VoiceForm Data API - Environment Setup")
    print("=" * 50)
    
    # Define paths
    current_dir = Path(__file__).parent
    env_example = current_dir / "env.example"
    env_file = current_dir / ".env"
    
    print(f"📂 Working directory: {current_dir}")
    print(f"📄 Template file: {env_example}")
    print(f"🎯 Target file: {env_file}")
    
    # Check if env.example exists
    if not env_example.exists():
        print(f"❌ Template file not found: {env_example}")
        print("   Please make sure you're in the correct directory.")
        return
    
    # Check if .env already exists
    if env_file.exists():
        print(f"⚠️  .env file already exists: {env_file}")
        response = input("   Do you want to overwrite it? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("✅ Setup cancelled - keeping existing .env file")
            return
    
    try:
        # Copy env.example to .env
        shutil.copy2(env_example, env_file)
        print(f"✅ Created .env file: {env_file}")
        
        # Show the content
        print(f"\n📝 Current .env configuration:")
        print("-" * 30)
        with open(env_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                print(f"{line_num:2d}: {line.rstrip()}")
        print("-" * 30)
        
        print(f"\n💡 Next steps:")
        print(f"   1. Edit the .env file with your actual API credentials")
        print(f"   2. Replace 'your-actual-api-key' with your real API key")
        print(f"   3. Replace 'your-actual-api-url' with your real API URL")
        print(f"   4. Save the file")
        print(f"   5. Run examples: python -m conversion_subnet.data_api.examples.example")
        
        print(f"\n🧪 Test your setup:")
        print(f"   python -m conversion_subnet.data_api.examples.fetch_data_example")
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return
    
    print(f"\n✅ Setup completed successfully!")

if __name__ == '__main__':
    main() 