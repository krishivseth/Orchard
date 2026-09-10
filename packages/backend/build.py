#!/usr/bin/env python3
"""
Build script to bundle the Orchard backend using PyInstaller.
This creates a standalone executable for macOS.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    print("Building Orchard Backend with PyInstaller...")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    
    # Output directory for the bundled executable
    output_dir = project_root / "packages" / "frontend" / "resources" / "backend"
    
    # Clean previous build
    dist_dir = script_dir / "dist"
    build_dir = script_dir / "build"
    spec_file = script_dir / "main.spec"
    
    for path in [dist_dir, build_dir, spec_file]:
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
    
    # PyInstaller command
    pyinstaller_args = [
        "pyinstaller",
        "--onefile",  # Single executable file
        "--name", "orchard-backend",  # Executable name
        "--clean",  # Clean cache
        "--noconfirm",  # Don't ask for confirmation
        # Hidden imports that PyInstaller might miss
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "websockets",
        "--hidden-import", "websockets.legacy",
        "--hidden-import", "websockets.legacy.server",
        "--hidden-import", "fastapi",
        "--hidden-import", "pydantic",
        "--hidden-import", "httpx",
        "--hidden-import", "loguru",
        "--hidden-import", "shared",
        "--hidden-import", "shared.types",
        "--hidden-import", "llama_sharding",
        "--collect-all", "fastapi",
        "--collect-all", "uvicorn",
        "--collect-all", "websockets",
        "--paths", str(script_dir),
        "main.py"
    ]
    
    print(f"Running PyInstaller with args: {' '.join(pyinstaller_args)}")
    
    # Change to backend directory
    os.chdir(script_dir)
    
    # Run PyInstaller
    result = subprocess.run(pyinstaller_args, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("PyInstaller failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    
    print("PyInstaller completed successfully!")
    
    # Copy the executable to the resources directory
    executable = dist_dir / "orchard-backend"
    
    if not executable.exists():
        print(f"Error: Executable not found at {executable}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy executable
    destination = output_dir / "orchard-backend"
    shutil.copy2(executable, destination)
    
    # Make it executable
    os.chmod(destination, 0o755)
    
    print(f"Backend executable copied to: {destination}")
    print("Build complete!")

if __name__ == "__main__":
    main()

