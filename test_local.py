#!/usr/bin/env python3
"""
Local testing script for CMS pipeline.
Run this to test your code before deploying to Azure.

Usage:
    python test_local.py
"""

import sys
import logging
from master import run_cms_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cms_pipeline_test.log')
    ]
)

def main():
    """Run the CMS pipeline locally for testing."""
    print("=" * 60)
    print("CMS Pipeline Local Test")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Connect to Azure Blob Storage")
    print("  2. Process CMS data")
    print("  3. Run geocoding (if needed)")
    print("  4. Calculate distances")
    print("  5. Run ML predictions")
    print("  6. Save outputs to blob storage")
    print("\n" + "=" * 60)
    
    try:
        logging.info("Starting CMS pipeline test...")
        run_cms_pipeline()
        logging.info("Pipeline completed successfully!")
        print("\n" + "=" * 60)
        print("✓ SUCCESS: Pipeline completed without errors")
        print("=" * 60)
        print("\nCheck blob storage for outputs:")
        print("  - cms_master.csv")
        print("  - cms_dedupe.csv")
        print("  - cms_visuals.csv")
        print("  - geocode_data.csv")
        print("  - inter_hospital_distances.csv")
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print("\n" + "=" * 60)
        print("✗ FAILED: Pipeline encountered errors")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        print("\nSee cms_pipeline_test.log for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
