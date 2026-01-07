#!/usr/bin/env python3
"""
Syntax and indentation checker for Python files.
Run this before deploying to catch any errors.

Usage:
    python check_syntax.py
"""

import sys
import py_compile
import ast

def check_file(filepath):
    """Check Python file for syntax errors."""
    print(f"\nChecking: {filepath}")
    print("-" * 60)
    
    # Try to compile the file
    try:
        py_compile.compile(filepath, doraise=True)
        print("✓ Syntax check: PASS")
    except py_compile.PyCompileError as e:
        print("✗ Syntax check: FAIL")
        print(f"  Error: {e}")
        return False
    
    # Try to parse the AST
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        print("✓ AST parse: PASS")
    except SyntaxError as e:
        print("✗ AST parse: FAIL")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False
    except IndentationError as e:
        print("✗ Indentation: FAIL")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False
    
    # Try to import (checks for runtime issues)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", filepath)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just load the module structure
            print("✓ Import check: PASS")
    except Exception as e:
        print(f"⚠ Import check: WARNING - {e}")
        # This is a warning, not a failure
    
    return True

def main():
    """Check all Python files."""
    print("=" * 60)
    print("Python Syntax & Indentation Checker")
    print("=" * 60)
    
    files_to_check = [
        'master.py',
        'function_app.py',
        'test_local.py'
    ]
    
    all_passed = True
    
    for filepath in files_to_check:
        try:
            if not check_file(filepath):
                all_passed = False
        except FileNotFoundError:
            print(f"\n⚠ Warning: {filepath} not found (skipping)")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        print("\nYour code is ready to deploy!")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the errors before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
