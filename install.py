#!/usr/bin/env python3
"""
Installation script for the Generative AI Terrain Prototype
Helps install dependencies and test the installation.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_package(package, description=""):
    """Install a Python package using pip."""
    print(f"Installing {package}...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def install_requirements():
    """Install packages from requirements.txt."""
    print("\nInstalling core requirements...")
    
    if os.path.exists("requirements.txt"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Core requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def install_optional_packages():
    """Install optional packages for advanced features."""
    print("\nInstalling optional packages for advanced features...")
    
    optional_packages = [
        "diffusers",
        "transformers", 
        "accelerate"
    ]
    
    success_count = 0
    for package in optional_packages:
        if install_package(package):
            success_count += 1
    
    if success_count == len(optional_packages):
        print("✅ All optional packages installed successfully")
        print("   Full diffusion model support enabled!")
    elif success_count > 0:
        print(f"⚠️  {success_count}/{len(optional_packages)} optional packages installed")
        print("   Basic functionality will work, but diffusion models may not be available")
    else:
        print("❌ No optional packages installed")
        print("   Only basic terrain generation will be available")
    
    return success_count

def test_imports():
    """Test if key packages can be imported."""
    print("\nTesting package imports...")
    
    test_packages = [
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch")
    ]
    
    success_count = 0
    for package, name in test_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
            success_count += 1
        except ImportError:
            print(f"❌ {name} import failed")
    
    # Test optional packages
    try:
        from diffusers import StableDiffusionControlNetPipeline
        print("✅ Diffusers imported successfully")
        success_count += 1
    except ImportError:
        print("⚠️  Diffusers not available (optional)")
    
    return success_count

def run_basic_test():
    """Run a basic test to verify functionality."""
    print("\nRunning basic functionality test...")
    
    try:
        # Import the main module
        from terrain_prototype import prompt_to_heightmap_gan
        
        # Test basic generation
        heightmap = prompt_to_heightmap_gan("test terrain", size=64)
        
        if heightmap is not None and heightmap.shape == (64, 64):
            print("✅ Basic terrain generation test passed")
            return True
        else:
            print("❌ Basic terrain generation test failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main installation function."""
    print("=" * 60)
    print("Generative AI Terrain Prototype - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease upgrade Python and try again.")
        return
    
    # Install core requirements
    if not install_requirements():
        print("\nFailed to install core requirements.")
        print("Please check your pip installation and try again.")
        return
    
    # Install optional packages
    install_optional_packages()
    
    # Test imports
    import_success = test_imports()
    
    # Run basic test
    test_success = run_basic_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    
    if import_success >= 4 and test_success:
        print("🎉 Installation completed successfully!")
        print("\nYou can now run:")
        print("  python terrain_prototype.py    # Interactive prototype")
        print("  python demo.py                 # Demo examples")
        print("  python test_terrain.py         # Run tests")
        
        print("\nFor the best experience, make sure you have:")
        print("  - A CUDA-compatible GPU (optional, for faster diffusion)")
        print("  - At least 4GB RAM (8GB+ recommended for diffusion models)")
        
    else:
        print("⚠️  Installation completed with issues.")
        print("\nBasic functionality should work, but some features may be limited.")
        print("You can still run:")
        print("  python terrain_prototype.py    # Basic prototype")
        print("  python test_terrain.py         # Test functionality")
        
        print("\nTo resolve issues:")
        print("  - Check error messages above")
        print("  - Ensure pip is up to date: python -m pip install --upgrade pip")
        print("  - Try installing packages individually if needed")
    
    print("\nFor help, check the README.md file or run the test script.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during installation: {e}")
        print("Please check the error message and try again.")
