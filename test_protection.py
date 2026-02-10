#!/usr/bin/env python3
"""
Test script for protection methods using FlowEdit YAML config
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path so we can import src modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_protection_test():
    """Run protection tests for all three methods"""
    
    # Paths
    config_file = "modules/FlowEdit/edits.yaml"
    output_base = "results/protection_test"
    
    # Methods to test
    methods = ["pid", "diff_protect"]  # Start with simpler methods first
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        print("Please make sure the FlowEdit module is properly set up.")
        return False
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()} Protection Method")
        print(f"{'='*60}")
        
        output_dir = os.path.join(output_base, method)
        
        # Build command
        cmd = [
            sys.executable, "scripts/run_protection.py",
            "--config", config_file,
            "--output_dir", output_dir,
            "--protection_method", method,
            "--protection_model", "sd1.4",
            "--evaluate"
        ]
        
        # Add method-specific parameters for faster testing
        if method == "atk_pdm":
            cmd.extend(["--optim_steps", "10"])  # Reduced for faster testing
        elif method == "diff_protect":
            cmd.extend(["--attack_mode", "mist", "--epsilon", "8.0"])
        elif method == "pid":
            cmd.extend(["--max_train_steps", "20", "--eps", "8.0"])  # Reduced for faster testing
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            success_count += 1
            print(f"✅ {method} test PASSED")
        except subprocess.CalledProcessError as e:
            print(f"❌ {method} test FAILED")
            print("STDOUT:")
            print(e.stdout)
            print("STDERR:")
            print(e.stderr)
        except Exception as e:
            print(f"❌ {method} test FAILED with exception: {e}")
        
        total_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {success_count}/{total_count}")
    if success_count == total_count:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_protection_test()
    sys.exit(0 if success else 1)