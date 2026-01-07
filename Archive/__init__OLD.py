"""
CMS Azure Function
This module will process and refresh dataset quarterly.
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from geopy.distance import great_circle
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import math
import statistics
import requests
import datetime as dt
from datetime import datetime, timedelta
import time
import io
import pickle
import hashlib
import re
from cah_functions import identify_potential_cah, calculate_hospital_distances

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger('cms_function')

# Get current year
current_year = datetime.now().year

# Global variables to cache data in memory
_alpha_data = None
_numeric_data = None
_date_df = None
_names_df = None
_zip_df = None
_address_df = None
_city_df = None
_states_df = None
_ccn_df = None
_control_df = None
_network_df = None

# Global variables for financial data caching
_netIncome_df = None
_interest_df = None
_taxes_df = None
_depreciation_df = None
_rent_df = None
_ebitda_df = None

# Global variables for additional financial metrics caching
_liabilities_df = None
_longLiabilities_df = None
_operatingCosts_df = None
_otherCosts_df = None
_assets_df = None
_reimburse_df = None
_receivables_df = None
_cash_df = None
_provider_df = None

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def load_alpha_data(use_cache=True):
    """
    Load ALPHA data from either local files or Azure Blob Storage and combine into a single DataFrame.
    
    Args:
        use_cache (bool): If True, use cached data if available. If False, always load from source files.
                         Set to False for Azure Function deployment to ensure fresh data.
    
    Returns:
        pandas.DataFrame: Combined ALPHA data from all available years
    """
    global _alpha_data
    
    # If data is already loaded in memory and caching is enabled, return it
    if use_cache and _alpha_data is not None:
        logger.info("Using cached ALPHA data from memory")
        return _alpha_data
    
    # Check if data is cached on disk and caching is enabled
    cache_file = os.path.join(CACHE_DIR, 'alpha_data.pkl')
    if use_cache and os.path.exists(cache_file):
        try:
            logger.info("Loading ALPHA data from disk cache")
            with open(cache_file, 'rb') as f:
                _alpha_data = pickle.load(f)
            logger.info(f"Successfully loaded cached data with {len(_alpha_data):,} records")
            return _alpha_data
        except Exception as e:
            logger.warning(f"Failed to load cached data: {str(e)}. Will load from source files.")
    
    # Load from local directory only
    logger.info("Loading ALPHA data from local files")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alpha_files = []
    alpha_pattern = r'HOSP10_?(\d{4})_ALPHA\.CSV'
    for filename in os.listdir(script_dir):
        match = re.match(alpha_pattern, filename, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            file_path = os.path.join(script_dir, filename)
            alpha_files.append((year, file_path, True))  # True indicates local file
            logger.info(f"Found local ALPHA file for year {year}: {filename}")
    
    if not alpha_files:
        logger.error("No ALPHA files found. Please check the file location or naming convention.")
        raise FileNotFoundError("No ALPHA files found")
    
    # Sort by year
    alpha_files.sort()
    logger.info(f"Found {len(alpha_files)} ALPHA files for years: {[year for year, _, _ in alpha_files]}")
    
    # Load and combine data from all years
    all_data = []
    azure_client = None
    for year, file_path, is_local in alpha_files:
        logger.info(f"Reading data for year {year}")
        
        try:
            if is_local:
                # Read local CSV file
                df = pd.read_csv(file_path, header=None)
            else:
                # Read CSV from Azure blob
                if azure_client is None:
                    azure_client = None
                df = azure_client.read_csv_from_blob(file_path, header=None)
            
            # Rename columns to standard names
            df.columns = ['Provider Number', 'Worksheet Code', 'Line Number', 'Column Number', 'Value']
            
            # Add Form ID and Year columns
            df['Form ID'] = 'HOSP10'
            df['Year'] = year
            
            all_data.append(df)
            logger.info(f"Successfully processed data for year {year} with {len(df):,} records")
            
        except Exception as e:
            logger.error(f"Error processing data for year {year}: {str(e)}")
            # Continue with next year if one fails
            continue
    
    # Combine all years into a single DataFrame
    alpha_data = pd.concat(all_data, ignore_index=True)
    
    # Store in global variable for future use
    _alpha_data = alpha_data
    
    # Cache to disk
    try:
        logger.info("Saving ALPHA data to disk cache")
        with open(cache_file, 'wb') as f:
            pickle.dump(alpha_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Successfully saved data to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save data to disk cache: {str(e)}")
    
    return alpha_data

def extract_hospital_data(alpha_data, use_cache=True):
    """
    Extract hospital information datasets from alpha data.
    
    Args:
        alpha_data (pandas.DataFrame): The ALPHA data to extract hospital information from
        use_cache (bool): If True, use cached data if available. If False, always extract from alpha_data.
                         Set to False for Azure Function deployment to ensure fresh data.
    
    Returns:
        tuple: Multiple DataFrames containing extracted hospital information
    """
    global _date_df, _names_df, _zip_df, _address_df, _city_df, _states_df, _ccn_df, _control_df, _network_df
    
    # If data is already extracted and in memory and caching is enabled, return it
    if use_cache and _date_df is not None and _names_df is not None:
        logger.info("Using cached hospital datasets from memory")
        return _date_df, _names_df, _zip_df, _address_df, _city_df, _states_df, _ccn_df, _control_df, _network_df
    
    # Check if data is cached on disk and caching is enabled
    cache_file = os.path.join(CACHE_DIR, 'hospital_data.pkl')
    if use_cache and os.path.exists(cache_file):
        try:
            logger.info("Loading hospital datasets from disk cache")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                _date_df = cache_data['date_df']
                _names_df = cache_data['names_df']
                _zip_df = cache_data['zip_df']
                _address_df = cache_data['address_df']
                _city_df = cache_data['city_df']
                _states_df = cache_data['states_df']
                _ccn_df = cache_data.get('ccn_df')  # Use get() to handle case where ccn_df doesn't exist in older cache files
                _control_df = cache_data.get('control_df')  # Use get() to handle case where control_df doesn't exist in older cache files
                _network_df = cache_data.get('network_df')  # Use get() to handle case where network_df doesn't exist in older cache files
            logger.info(f"Successfully loaded cached hospital data")
            return _date_df, _names_df, _zip_df, _address_df, _city_df, _states_df, _ccn_df, _control_df, _network_df
        except Exception as e:
            logger.warning(f"Failed to load cached hospital data: {str(e)}. Will extract from alpha data.")
    
    logger.info("Creating hospital information datasets")
    
    # Get reporting start date
    logger.info("Extracting report start dates")
    date_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                             (alpha_data['Column Number'] == 100) & 
                             (alpha_data['Line Number'] == 2000)].copy()
    date_df = date_df.rename(columns = {'Value': 'Report Start Date'})
    date_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(date_df)} report start dates")
    
    # Get hospital names
    logger.info("Extracting hospital names")
    names_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                              (alpha_data['Column Number'] == 100) & 
                              (alpha_data['Line Number'] == 300)].copy()
    names_df = names_df.rename(columns = {'Value': 'Hospital Name'})
    names_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(names_df)} hospital names")
    
    # Get zip codes
    logger.info("Extracting zip codes")
    zip_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                            (alpha_data['Column Number'] == 300) & 
                            (alpha_data['Line Number'] == 200)].copy()
    zip_df = zip_df.rename(columns = {'Value': 'Zip Code'})
    zip_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(zip_df)} zip codes")
    
    # Get hospital addresses
    logger.info("Extracting hospital addresses")
    address_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                                (alpha_data['Column Number'] == 100) & 
                                (alpha_data['Line Number'] == 100)].copy()
    address_df = address_df.rename(columns = {'Value': 'Address'})
    address_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(address_df)} addresses")
    
    # Get hospital cities
    logger.info("Extracting hospital cities")
    city_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                             (alpha_data['Column Number'] == 100) & 
                             (alpha_data['Line Number'] == 200)].copy()
    city_df = city_df.rename(columns = {'Value': 'City'})
    city_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(city_df)} cities")
    
    # Get state names
    logger.info("Extracting state names")
    states_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                               (alpha_data['Column Number'] == 200) & 
                               (alpha_data['Line Number'] == 200)].copy()
    states_df = states_df.rename(columns = {'Value': 'State'})
    states_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(states_df)} states")
    
    # Get provider type
    logger.info("Extracting provider types")
    ccn_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                            (alpha_data['Column Number'] == 200) & 
                            (alpha_data['Line Number'] == 300)].copy()
    ccn_df = ccn_df.rename(columns = {'Value': 'CCN'})
    ccn_df['Facility Code'] = ccn_df['CCN'].str[-4:-2]
    
    # Map only facility code '13' to 'CAH', leave others as NaN
    # Map Critical Access Hospital to a consistent, descriptive label
    ccn_df['Facility Type'] = np.where(
        ccn_df['Facility Code'] == '13',
        'Critical Access Hospital',
        np.nan
    )
    
    ccn_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(ccn_df)} provider records")
    _ccn_df = ccn_df
    
    # Get type of control
    logger.info("Extracting hospital control types")
    control_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                               (alpha_data['Column Number'] == 100) & 
                               (alpha_data['Line Number'] == 2100)].copy()
    control_df = control_df.rename(columns = {'Value': 'Control Code'})
    
    # Define control type mapping dictionary
    control_dict = {
        '1':  'Non-Profit', 
        '2':  'Non-Profit', 
        '3':  'Private',
        '4':  'Private',
        '5':  'Private',
        '6':  'Private',
        '7':  'Government - Federal',
        '8':  'Government - City-County',
        '9':  'Government - County',
        '10': 'Government - State',
        '11': 'Government',
        '12': 'Government - City',
        '13': 'Government'    
    }
    
    # Map control codes to control types
    control_df['Type of Control'] = control_df['Control Code'].map(control_dict)
    control_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(control_df)} hospital control types")
    _control_df = control_df
    
    # Get hospital status (independent or network)
    logger.info("Extracting hospital network status")
    network_df = alpha_data.loc[(alpha_data['Worksheet Code'] == 'S200001') & 
                               (alpha_data['Column Number'] == 100) & 
                               (alpha_data['Line Number'] == 14100)].copy()
    network_df = network_df.rename(columns = {'Value': 'Network'})
    network_df['Network'] = 'Part of Network'
    network_df.reset_index(drop = True, inplace = True)
    logger.info(f"Found {len(network_df)} hospitals in networks")
    _network_df = network_df
    
    # Store in global variables for future use
    _date_df = date_df
    _names_df = names_df
    _zip_df = zip_df
    _address_df = address_df
    _city_df = city_df
    _states_df = states_df
    
    # Cache to disk
    try:
        logger.info("Saving hospital datasets to disk cache")
        cache_data = {
            'date_df': date_df,
            'names_df': names_df,
            'zip_df': zip_df,
            'address_df': address_df,
            'city_df': city_df,
            'states_df': states_df,
            'ccn_df': ccn_df,
            'control_df': control_df,
            'network_df': network_df
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Successfully saved hospital data to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save hospital data to disk cache: {str(e)}")
    
    return date_df, names_df, zip_df, address_df, city_df, states_df, ccn_df, control_df, network_df

def main(use_cache=True):
    """
    Main entry point for local execution.
    
    Args:
        use_cache (bool): If True, use cached data if available. If False, always load from source files.
                         Set to False for Azure Function deployment to ensure fresh data.
    
    Returns:
        tuple: The loaded data and extracted metrics
    """
    logger.info("Starting CMS data processing")
    
    try:
        # Load ALPHA data from source files (uses cached data if use_cache is True)
        alpha_data = load_alpha_data(use_cache=use_cache)
        
        # Log summary information
        logger.info(f"Loaded {len(alpha_data):,} records from {len(alpha_data['Year'].unique())} years")
        logger.info(f"Year range: {alpha_data['Year'].min()} to {alpha_data['Year'].max()}")
        
        # Check unique years available in the data
        unique_years = alpha_data['Year'].unique()
        logger.info(f"Unique years in dataset: {sorted(unique_years)}")
        
        # Load numeric data from source files (uses cached data if use_cache is True)
        numeric_data = load_numeric_data(use_cache=use_cache)

        # Build comprehensive dataset
        logger.info("Assembling comprehensive CMS dataset")
        cms_df = assemble_comprehensive_dataset(alpha_data, numeric_data)

        # Attach geocodes and radians
        logger.info("Merging geocodes and adding radians")
        cms_df = attach_geocodes(cms_df)

        # Compute distances (<=35mi, exclude <1mi) with caching and save CSV snapshot
        logger.info("Computing and caching neighbor distances")
        distances_df = compute_distances_and_cache(cms_df, max_distance=35.0, min_distance=1.0, use_cache=use_cache)

        # Compute Potential CAH
        logger.info("Computing Potential_CAH flags")
        cms_df = compute_potential_cah(cms_df, distances_df, beds_limit=25)

        # Add financial heuristic tags
        logger.info("Tagging financial heuristics")
        cms_df = tag_financial_heuristics(cms_df)

        # Train and score Build_Odds model
        logger.info("Training and scoring Build_Odds model")
        try:
            cms_df = add_build_odds(cms_df)
        except Exception as e:
            logger.warning(f"Build_Odds step skipped: {e}")

        # Export stable CSV locally
        output_file = os.path.join(os.path.dirname(__file__), "cms_data.csv")
        cms_df.to_csv(output_file, index=False)
        logger.info(f"Saved CMS data to {output_file}")

        logger.info("Processing complete")
        return cms_df, distances_df
        
    except Exception as e:
        logger.error(f"Error in CMS processing: {str(e)}")
        raise

def load_numeric_data(use_cache=True):
    """
    Load numeric data from NMRC CSV files and combine into a single DataFrame.
    
    Args:
        use_cache (bool): If True, use cached data if available. If False, always load from source files.
    
    Returns:
        pandas.DataFrame: Combined numeric data from all available years
    """
    global _numeric_data
    
    # If data is already loaded in memory, return it
    if _numeric_data is not None and use_cache:
        logger.info("Using in-memory cached numeric data")
        return _numeric_data
    
    # Check if cached data exists on disk
    cache_file = os.path.join('cache', 'numeric_data.pkl')
    if os.path.exists(cache_file) and use_cache:
        try:
            logger.info("Loading numeric data from disk cache")
            with open(cache_file, 'rb') as f:
                _numeric_data = pickle.load(f)
            logger.info(f"Successfully loaded cached data with {len(_numeric_data):,} records")
            return _numeric_data
        except Exception as e:
            logger.warning(f"Failed to load numeric data from disk cache: {str(e)}")
    
    # Load data from source files
    logger.info("Loading numeric data from source files")
    
    # Initialize empty DataFrame to hold all numeric data
    _numeric_data = pd.DataFrame()
    
    # Process local files only
    try:
        logger.info("Looking for NMRC files in local directory")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        all_files = os.listdir(script_dir)
        nmrc_files = []
        
        for filename in all_files:
            if re.search(r'HOSP10_\d{4}_NMRC\.CSV', filename, re.IGNORECASE):
                year_match = re.search(r'HOSP10_(\d{4})_NMRC\.CSV', filename, re.IGNORECASE)
                if year_match:
                    year = int(year_match.group(1))
                    nmrc_files.append((year, os.path.join(script_dir, filename)))
        
        logger.info(f"Found {len(nmrc_files)} NMRC files in local directory")
        
        # Process each local file
        for year, filepath in nmrc_files:
            try:
                logger.info(f"Reading numeric data for year {year} from local file: {os.path.basename(filepath)}")
                df = pd.read_csv(filepath, header=None)
                
                # Process the data
                df = df.rename(columns={0: 'Report Record Number', 1: 'Worksheet Code', 
                                       2: 'Line Number', 3: 'Column Number', 4: 'Value'})
                df['Form ID'] = df['Worksheet Code'] + ',' + df['Line Number'].astype(str) + ',' + df['Column Number'].astype(str)
                df['Year'] = year
                
                # Append to combined DataFrame
                _numeric_data = pd.concat([_numeric_data, df])
                _numeric_data.reset_index(drop=True, inplace=True)
                
                logger.info(f"Successfully processed numeric data for year {year} with {len(df):,} records")
            except Exception as e:
                logger.error(f"Error processing numeric data for year {year}: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing local files: {str(e)}")
    
    # Log summary and cache data
    if len(_numeric_data) > 0:
        logger.info(f"Loaded {len(_numeric_data):,} records from {len(_numeric_data['Year'].unique())} years")
        
        # Cache data to disk if we have data and caching is enabled
        if use_cache:
            try:
                # Ensure cache directory exists
                os.makedirs('cache', exist_ok=True)
                
                logger.info("Saving numeric data to disk cache")
                with open(cache_file, 'wb') as f:
                    pickle.dump(_numeric_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Successfully saved numeric data to disk cache")
            except Exception as e:
                logger.warning(f"Failed to save numeric data to disk cache: {str(e)}")
    else:
        logger.warning("No numeric data was loaded")
    
    return _numeric_data

# Global variables to cache hospital metrics
_beds_df = None
_discharge_df = None
_rural_df = None
_charity_df = None
_uncompensated_df = None

def extract_hospital_metrics(numeric_data, use_cache=True):
    """
    Extract specific hospital metrics from the numeric data.
    
    Args:
        numeric_data (pandas.DataFrame): The numeric data to extract metrics from
        use_cache (bool): If True, use cached data if available. If False, always extract from source data.
    
    Returns:
        tuple: Multiple DataFrames containing extracted metrics (beds_df, discharge_df, rural_df, charity_df, uncompensated_df)
    """
    global _beds_df, _discharge_df, _rural_df, _charity_df, _uncompensated_df
    
    # Check if data is already loaded in memory
    if use_cache and _beds_df is not None and _discharge_df is not None and _rural_df is not None and _charity_df is not None and _uncompensated_df is not None:
        logger.info("Using in-memory cached hospital metrics")
        return _beds_df, _discharge_df, _rural_df, _charity_df, _uncompensated_df
    
    # Check if cached data exists on disk
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    if use_cache:
        try:
            # Try to load all metrics from disk cache
            all_loaded = True
            
            # Load beds data
            beds_cache = os.path.join(cache_dir, 'beds_data.pkl')
            if os.path.exists(beds_cache):
                with open(beds_cache, 'rb') as f:
                    _beds_df = pickle.load(f)
                logger.info(f"Loaded {len(_beds_df)} bed records from cache")
            else:
                all_loaded = False
            
            # Load discharge data
            discharge_cache = os.path.join(cache_dir, 'discharge_data.pkl')
            if os.path.exists(discharge_cache):
                with open(discharge_cache, 'rb') as f:
                    _discharge_df = pickle.load(f)
                logger.info(f"Loaded {len(_discharge_df)} discharge records from cache")
            else:
                all_loaded = False
            
            # Load rural/urban data
            rural_cache = os.path.join(cache_dir, 'rural_data.pkl')
            if os.path.exists(rural_cache):
                with open(rural_cache, 'rb') as f:
                    _rural_df = pickle.load(f)
                logger.info(f"Loaded {len(_rural_df)} rural/urban records from cache")
            else:
                all_loaded = False
            
            # Load charity data
            charity_cache = os.path.join(cache_dir, 'charity_data.pkl')
            if os.path.exists(charity_cache):
                with open(charity_cache, 'rb') as f:
                    _charity_df = pickle.load(f)
                logger.info(f"Loaded {len(_charity_df)} charity cost records from cache")
            else:
                all_loaded = False
            
            # Load uncompensated care data
            uncompensated_cache = os.path.join(cache_dir, 'uncompensated_data.pkl')
            if os.path.exists(uncompensated_cache):
                with open(uncompensated_cache, 'rb') as f:
                    _uncompensated_df = pickle.load(f)
                logger.info(f"Loaded {len(_uncompensated_df)} uncompensated care records from cache")
            else:
                all_loaded = False
            
            # If all data was loaded from cache, return it
            if all_loaded:
                logger.info("Successfully loaded all hospital metrics from cache")
                return _beds_df, _discharge_df, _rural_df, _charity_df, _uncompensated_df
            else:
                logger.info("Some hospital metrics not found in cache, extracting from source data")
        except Exception as e:
            logger.warning(f"Failed to load hospital metrics from cache: {str(e)}")
    
    logger.info("Extracting hospital metrics from numeric data")
    
    try:
        # Get number beds
        logger.info("Extracting number of beds data")
        _beds_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S300001') & 
                                  (numeric_data['Column Number'].astype(str) == '00200') & 
                                  (numeric_data['Line Number'] == 100)].copy()
        _beds_df = _beds_df.rename(columns={'Value': 'Number Beds'})
        _beds_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_beds_df)} records with bed count information")
        
        # Get discharges
        logger.info("Extracting discharge data")
        _discharge_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'E10A182') & 
                                       (numeric_data['Column Number'].astype(str) == '00100') & 
                                       (numeric_data['Line Number'] == 100)].copy()
        _discharge_df = _discharge_df.rename(columns={'Value': 'Discharges'})
        _discharge_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_discharge_df)} records with discharge information")
        
        # Get rural vs. urban
        logger.info("Extracting rural vs. urban status")
        _rural_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S200001') & 
                                   (numeric_data['Column Number'].astype(str) == '00100') & 
                                   (numeric_data['Line Number'] == 2600)].copy()
        _rural_df = _rural_df.rename(columns={'Value': 'Rural vs. Urban'})
        _rural_df['Rural vs. Urban'] = np.where(_rural_df['Rural vs. Urban'] == 1, 'Urban', 
                                            np.where(_rural_df['Rural vs. Urban'] == 2, 'Rural', pd.NA))
        _rural_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_rural_df)} records with rural/urban status")
        
        # Get cost of charity - NOTE: INCLUDES UNINSURED / INSURED PATIENTS BUT THIS DOES NOT MATCH THE PUF DATA WHICH ONLY REPORTS UNINSURED PATIENTS
        logger.info("Extracting charity cost data")
        _charity_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S100000') & 
                                     (numeric_data['Column Number'].astype(str) == '00300') & 
                                     (numeric_data['Line Number'] == 2300)].copy()
        _charity_df = _charity_df.rename(columns={'Value': 'Cost of Charity'})
        _charity_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_charity_df)} records with charity cost information")
        
        # Get cost of uncompensated care
        logger.info("Extracting uncompensated care cost data")
        _uncompensated_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S100000') & 
                                           (numeric_data['Column Number'].astype(str) == '00100') & 
                                           (numeric_data['Line Number'] == 3000)].copy()
        _uncompensated_df = _uncompensated_df.rename(columns={'Value': 'Cost of Uncompensated Care'})
        _uncompensated_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_uncompensated_df)} records with uncompensated care cost information")
        
        # Cache data to disk if caching is enabled
        if use_cache:
            try:
                # Save beds data
                with open(os.path.join(cache_dir, 'beds_data.pkl'), 'wb') as f:
                    pickle.dump(_beds_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save discharge data
                with open(os.path.join(cache_dir, 'discharge_data.pkl'), 'wb') as f:
                    pickle.dump(_discharge_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save rural/urban data
                with open(os.path.join(cache_dir, 'rural_data.pkl'), 'wb') as f:
                    pickle.dump(_rural_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save charity data
                with open(os.path.join(cache_dir, 'charity_data.pkl'), 'wb') as f:
                    pickle.dump(_charity_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save uncompensated care data
                with open(os.path.join(cache_dir, 'uncompensated_data.pkl'), 'wb') as f:
                    pickle.dump(_uncompensated_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.info("Successfully cached hospital metrics to disk")
            except Exception as e:
                logger.warning(f"Failed to cache hospital metrics to disk: {str(e)}")
        
        return _beds_df, _discharge_df, _rural_df, _charity_df, _uncompensated_df
        
    except Exception as e:
        logger.error(f"Error extracting hospital metrics: {str(e)}")
        raise

# Global variables to cache financial data
_netIncome_df = None
_interest_df = None
_taxes_df = None
_depreciation_df = None
_rent_df = None
_ebitda_df = None

# Global variables for additional financial metrics
_liabilities_df = None
_longLiabilities_df = None
_operatingCosts_df = None
_otherCosts_df = None
_assets_df = None
_reimburse_df = None
_receivables_df = None
_cash_df = None
_provider_df = None

def extract_financial_data(numeric_data, use_cache=True):
    """
    Extract financial data from numeric dataset including net income, interest expense,
    tax expense, depreciation/amortization expense, and rent costs.
    
    Args:
        numeric_data (pandas.DataFrame): The numeric data to extract financial data from
        use_cache (bool): If True, use cached data if available. If False, always extract from source data.
    
    Returns:
        tuple: Multiple DataFrames containing extracted financial data
    """
    global _netIncome_df, _interest_df, _taxes_df, _depreciation_df, _rent_df
    
    # Check if data is already loaded in memory
    if use_cache and _netIncome_df is not None and _interest_df is not None and _taxes_df is not None and _depreciation_df is not None and _rent_df is not None:
        logger.info("Using in-memory cached financial data")
        return _netIncome_df, _interest_df, _taxes_df, _depreciation_df, _rent_df
    
    # Check if cached data exists on disk
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    if use_cache:
        try:
            # Try to load all financial data from disk cache
            all_loaded = True
            
            # Load net income data
            netIncome_cache = os.path.join(cache_dir, 'netIncome_data.pkl')
            if os.path.exists(netIncome_cache):
                with open(netIncome_cache, 'rb') as f:
                    _netIncome_df = pickle.load(f)
                logger.info(f"Loaded {len(_netIncome_df)} net income records from cache")
            else:
                all_loaded = False
            
            # Load interest data
            interest_cache = os.path.join(cache_dir, 'interest_data.pkl')
            if os.path.exists(interest_cache):
                with open(interest_cache, 'rb') as f:
                    _interest_df = pickle.load(f)
                logger.info(f"Loaded {len(_interest_df)} interest expense records from cache")
            else:
                all_loaded = False
            
            # Load taxes data
            taxes_cache = os.path.join(cache_dir, 'taxes_data.pkl')
            if os.path.exists(taxes_cache):
                with open(taxes_cache, 'rb') as f:
                    _taxes_df = pickle.load(f)
                logger.info(f"Loaded {len(_taxes_df)} tax expense records from cache")
            else:
                all_loaded = False
            
            # Load depreciation data
            depreciation_cache = os.path.join(cache_dir, 'depreciation_data.pkl')
            if os.path.exists(depreciation_cache):
                with open(depreciation_cache, 'rb') as f:
                    _depreciation_df = pickle.load(f)
                logger.info(f"Loaded {len(_depreciation_df)} depreciation expense records from cache")
            else:
                all_loaded = False
            
            # Load rent data
            rent_cache = os.path.join(cache_dir, 'rent_data.pkl')
            if os.path.exists(rent_cache):
                with open(rent_cache, 'rb') as f:
                    _rent_df = pickle.load(f)
                logger.info(f"Loaded {len(_rent_df)} rent expense records from cache")
            else:
                all_loaded = False
            
            # If all data was loaded from cache, return it
            if all_loaded:
                logger.info("Successfully loaded all financial data from cache")
                return _netIncome_df, _interest_df, _taxes_df, _depreciation_df, _rent_df
            else:
                logger.info("Some financial data not found in cache, extracting from source data")
        except Exception as e:
            logger.warning(f"Failed to load financial data from cache: {str(e)}")
    
    logger.info("Extracting financial data from numeric data")
    
    try:
        # Get net income
        logger.info("Extracting net income data")
        _netIncome_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G300000') & 
                                       (numeric_data['Column Number'].astype(str) == '00100') & 
                                       (numeric_data['Line Number'] == 2900)].copy()
        _netIncome_df = _netIncome_df.rename(columns={'Value': 'Net Income'})
        _netIncome_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_netIncome_df)} records with net income information")
        
        # Get interest expense
        logger.info("Extracting interest expense data")
        _interest_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & 
                                      (numeric_data['Column Number'].astype(str) == '01100') & 
                                      (numeric_data['Line Number'] == 300)].copy()
        _interest_df = _interest_df.rename(columns={'Value': 'Interest'})
        _interest_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_interest_df)} records with interest expense information")
        
        # Get tax expense
        logger.info("Extracting tax expense data")
        _taxes_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & 
                                   (numeric_data['Column Number'].astype(str) == '01300') & 
                                   (numeric_data['Line Number'] == 300)].copy()
        _taxes_df = _taxes_df.rename(columns={'Value': 'Taxes'})
        _taxes_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_taxes_df)} records with tax expense information")
        
        # Get depreciation / amortization expense
        logger.info("Extracting depreciation/amortization expense data")
        _depreciation_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & 
                                          (numeric_data['Column Number'].astype(str) == '00900') & 
                                          (numeric_data['Line Number'] == 300)].copy()
        _depreciation_df = _depreciation_df.rename(columns={'Value': 'Depreciation'})
        _depreciation_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_depreciation_df)} records with depreciation expense information")
        
        # Get rent costs
        logger.info("Extracting rent costs data")
        _rent_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & 
                                  (numeric_data['Column Number'].astype(str) == '01000') & 
                                  (numeric_data['Line Number'] == 300)].copy()
        _rent_df = _rent_df.rename(columns={'Value': 'Rent'})
        _rent_df.reset_index(drop=True, inplace=True)
        logger.info(f"Found {len(_rent_df)} records with rent costs information")
        
        # Cache data to disk if caching is enabled
        if use_cache:
            try:
                # Save net income data
                with open(os.path.join(cache_dir, 'netIncome_data.pkl'), 'wb') as f:
                    pickle.dump(_netIncome_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save interest data
                with open(os.path.join(cache_dir, 'interest_data.pkl'), 'wb') as f:
                    pickle.dump(_interest_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save taxes data
                with open(os.path.join(cache_dir, 'taxes_data.pkl'), 'wb') as f:
                    pickle.dump(_taxes_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save depreciation data
                with open(os.path.join(cache_dir, 'depreciation_data.pkl'), 'wb') as f:
                    pickle.dump(_depreciation_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save rent data
                with open(os.path.join(cache_dir, 'rent_data.pkl'), 'wb') as f:
                    pickle.dump(_rent_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.info("Successfully cached financial data to disk")
            except Exception as e:
                logger.warning(f"Failed to cache financial data to disk: {str(e)}")
        
        return _netIncome_df, _interest_df, _taxes_df, _depreciation_df, _rent_df
        
    except Exception as e:
        logger.error(f"Error extracting financial data: {str(e)}")
        raise

def calculate_ebitda(netIncome_df, interest_df, taxes_df, depreciation_df, rent_df=None, use_cache=True):
    """
    Calculate EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) from financial data.
    Optionally includes rent costs in the calculation (EBITDAR).
    
    Args:
        netIncome_df (pandas.DataFrame): DataFrame containing net income data
        interest_df (pandas.DataFrame): DataFrame containing interest expense data
        taxes_df (pandas.DataFrame): DataFrame containing tax expense data
        depreciation_df (pandas.DataFrame): DataFrame containing depreciation expense data
        rent_df (pandas.DataFrame, optional): DataFrame containing rent costs data
        use_cache (bool): If True, use cached data if available
    
    Returns:
        pandas.DataFrame: DataFrame containing EBITDA calculations
    """
    global _ebitda_df
    
    # Check if data is already loaded in memory
    if use_cache and _ebitda_df is not None:
        logger.info("Using in-memory cached EBITDA data")
        return _ebitda_df
    
    # Check if cached data exists on disk
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    ebitda_cache = os.path.join(cache_dir, 'ebitda_data.pkl')
    
    if use_cache and os.path.exists(ebitda_cache):
        try:
            with open(ebitda_cache, 'rb') as f:
                _ebitda_df = pickle.load(f)
            logger.info(f"Loaded {len(_ebitda_df)} EBITDA records from cache")
            return _ebitda_df
        except Exception as e:
            logger.warning(f"Failed to load EBITDA data from cache: {str(e)}")
    
    logger.info("Calculating EBITDA from financial data")
    
    try:
        # Calculate EBITDAR using the provided code
        # Merge all financial components using outer joins
        ebitda_df = interest_df[['Report Record Number', 'Interest', 'Year']].merge(
            taxes_df[['Report Record Number', 'Taxes', 'Year']], 
            on=['Report Record Number', 'Year'], 
            how='outer'
        )
        ebitda_df = ebitda_df.merge(
            depreciation_df[['Report Record Number', 'Depreciation', 'Year']], 
            on=['Report Record Number', 'Year'], 
            how='outer'
        )
        ebitda_df = ebitda_df.merge(
            rent_df[['Report Record Number', 'Rent', 'Year']], 
            on=['Report Record Number', 'Year'], 
            how='outer'
        )
        ebitda_df = ebitda_df.merge(
            netIncome_df[['Report Record Number', 'Net Income', 'Year']], 
            on=['Report Record Number', 'Year'], 
            how='outer'
        )
        
        # Fill missing values with 0
        ebitda_df = ebitda_df.fillna(0)
        
        # Calculate EBITDA and EBITDAR
        ebitda_df['EBITDA'] = ebitda_df['Net Income'] + ebitda_df['Interest'] + ebitda_df['Taxes'] + ebitda_df['Depreciation']
        ebitda_df['EBITDAR'] = ebitda_df['Net Income'] + ebitda_df['Interest'] + ebitda_df['Taxes'] + ebitda_df['Depreciation'] + ebitda_df['Rent']
        
        # Reset index
        ebitda_df.reset_index(drop=True, inplace=True)
        
        # Store in global variable
        _ebitda_df = ebitda_df
        
        logger.info(f"Calculated EBITDA for {len(_ebitda_df)} records")
        
        # Cache data to disk if caching is enabled
        if use_cache:
            try:
                with open(ebitda_cache, 'wb') as f:
                    pickle.dump(_ebitda_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Successfully cached EBITDA data to disk")
            except Exception as e:
                logger.warning(f"Failed to cache EBITDA data to disk: {str(e)}")
        
        return _ebitda_df
        
    except Exception as e:
        logger.error(f"Error calculating EBITDA: {str(e)}")
        raise

# ------------------------- CAH Pipeline Helpers -------------------------
def _safe_merge(left_df, right_df, right_column, on_column='Provider Number', how='outer'):
    """Safely merge a single column from right_df into left_df on a key."""
    try:
        if right_df is None or len(right_df) == 0:
            return left_df
        if right_column not in right_df.columns or on_column not in left_df.columns or on_column not in right_df.columns:
            return left_df
        subset = right_df[[on_column, right_column]].copy()
        result = left_df.merge(subset, on=on_column, how=how)
        return result
    except Exception as e:
        logger.warning(f"_safe_merge failed for column '{right_column}': {e}")
        return left_df

def assemble_comprehensive_dataset(alpha_data, numeric_data):
    """
    Assemble a comprehensive CMS dataset similar to export_hospital_data.py output,
    including key hospital info, metrics, and financials required for CAH logic.
    Returns cms_df DataFrame.
    """
    # Extract info and metrics
    date_df, names_df, zip_df, address_df, city_df, states_df, ccn_df, control_df, network_df = extract_hospital_data(alpha_data, use_cache=True)
    beds_df, discharge_df, rural_df, charity_df, uncompensated_df = extract_hospital_metrics(numeric_data, use_cache=True)

    # Attempt to load financials and additional metrics
    try:
        netIncome_df, interest_df, taxes_df, depreciation_df, rent_df = extract_financial_data(numeric_data, use_cache=True)
    except Exception:
        netIncome_df = interest_df = taxes_df = depreciation_df = rent_df = None
    try:
        ebitda_df = calculate_ebitda(netIncome_df, interest_df, taxes_df, depreciation_df, rent_df, use_cache=True) if netIncome_df is not None else None
    except Exception:
        ebitda_df = None
    try:
        addl = extract_additional_financial_metrics(numeric_data, use_cache=True)
        liabilities_df = addl.get('liabilities')
        longLiabilities_df = addl.get('long_liabilities')
        operatingCosts_df = addl.get('operating_costs')
        otherCosts_df = addl.get('other_costs')
        assets_df = addl.get('assets')
        reimburse_df = addl.get('reimbursements')
        receivables_df = addl.get('receivables')
        cash_df = addl.get('cash')
    except Exception:
        liabilities_df = longLiabilities_df = operatingCosts_df = otherCosts_df = assets_df = reimburse_df = receivables_df = cash_df = None

    # Base
    cms_df = names_df[['Provider Number', 'Hospital Name', 'Year']].copy()

    # Hospital info merges
    cms_df = _safe_merge(cms_df, states_df, 'State')
    cms_df = _safe_merge(cms_df, city_df, 'City')
    cms_df = _safe_merge(cms_df, address_df, 'Address')
    cms_df = _safe_merge(cms_df, zip_df, 'Zip Code')
    cms_df = _safe_merge(cms_df, date_df, 'Report Start Date')
    cms_df = _safe_merge(cms_df, ccn_df, 'CCN')
    if ccn_df is not None and 'Facility Type' in ccn_df.columns:
        cms_df = _safe_merge(cms_df, ccn_df, 'Facility Type')
    if control_df is not None:
        if 'Type of Control' in control_df.columns:
            cms_df = _safe_merge(cms_df, control_df, 'Type of Control')
        elif 'Control Code' in control_df.columns:
            cms_df = _safe_merge(cms_df, control_df, 'Control Code')
    cms_df = _safe_merge(cms_df, network_df, 'Network')

    # Metrics helpers
    def rn(df):
        return df.rename(columns={'Report Record Number': 'Provider Number'}) if df is not None and 'Report Record Number' in df.columns else df
    cms_df = _safe_merge(cms_df, rn(beds_df), 'Number Beds')
    cms_df = _safe_merge(cms_df, rn(discharge_df), 'Discharges')
    cms_df = _safe_merge(cms_df, rn(rural_df), 'Rural vs. Urban')
    cms_df = _safe_merge(cms_df, rn(charity_df), 'Cost of Charity')
    cms_df = _safe_merge(cms_df, rn(uncompensated_df), 'Cost of Uncompensated Care')

    # Financials
    if netIncome_df is not None:
        cms_df = _safe_merge(cms_df, rn(netIncome_df), 'Net Income')
    if ebitda_df is not None:
        cms_df = _safe_merge(cms_df, rn(ebitda_df), 'EBITDAR')
    if liabilities_df is not None:
        cms_df = _safe_merge(cms_df, rn(liabilities_df), 'Total Current Liabilities')
    if longLiabilities_df is not None:
        cms_df = _safe_merge(cms_df, rn(longLiabilities_df), 'Total Long-Term Liabilities')
    if cash_df is not None:
        cms_df = _safe_merge(cms_df, rn(cash_df), 'Cash')
    if operatingCosts_df is not None:
        cms_df = _safe_merge(cms_df, rn(operatingCosts_df), 'Operating Expense')
    if otherCosts_df is not None:
        cms_df = _safe_merge(cms_df, rn(otherCosts_df), 'Other Expense')
    if reimburse_df is not None:
        cms_df = _safe_merge(cms_df, rn(reimburse_df), 'Government Reimbursements')
    if receivables_df is not None:
        cms_df = _safe_merge(cms_df, rn(receivables_df), 'Net Receivables')
    if assets_df is not None:
        cms_df = _safe_merge(cms_df, rn(assets_df), 'Total Current Assets')

    # Independent check
    if 'Network' in cms_df.columns:
        cms_df['Check'] = cms_df['Network'].fillna('Independent')

    # Zip 5
    if 'Zip Code' in cms_df.columns:
        cms_df['Zip Code'] = cms_df['Zip Code'].astype(str).str.split('-').str[0].str.zfill(5)

    # Key rename
    cms_df = cms_df.rename(columns={'Provider Number': 'Report Record Number'})
    return cms_df

def attach_geocodes(cms_df, geocode_csv_path=None):
    """Merge Latitude/Longitude into cms_df using addresses_geocode.csv and add radians columns."""
    try:
        if geocode_csv_path is None:
            geocode_csv_path = os.path.join(os.path.dirname(__file__), 'addresses_geocode.csv')
        geo_df = pd.read_csv(geocode_csv_path)
        addr = cms_df.get('Address')
        city = cms_df.get('City')
        state = cms_df.get('State')
        zipc = cms_df.get('Zip Code')
        if addr is None or city is None or state is None or zipc is None:
            logger.warning('Missing address components for geocode merge; skipping geocodes merge')
            return cms_df
        key_series = addr.astype(str) + ', ' + city.astype(str).str.lower() + ', ' + state.astype(str).str.upper() + ', ' + zipc.astype(str).str.zfill(5)
        cms_df['Full Address'] = key_series
        cms_df = cms_df.merge(geo_df[['Full Address', 'Latitude', 'Longitude']], on='Full Address', how='left')
        if 'Latitude' in cms_df.columns and 'Longitude' in cms_df.columns:
            cms_df['Lat'] = np.radians(cms_df['Latitude'].astype('float64'))
            cms_df['Long'] = np.radians(cms_df['Longitude'].astype('float64'))
        return cms_df
    except Exception as e:
        logger.warning(f'attach_geocodes failed: {e}')
        return cms_df

def compute_distances_and_cache(cms_df, max_distance=35.0, min_distance=1.0, use_cache=True):
    """Compute pairwise distances within max_distance, exclude <min_distance, and cache the result to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_pkl = os.path.join(CACHE_DIR, f'distances_{int(max_distance)}mi.pkl')
    if use_cache and os.path.exists(cache_pkl):
        try:
            with open(cache_pkl, 'rb') as f:
                distances_df = pickle.load(f)
            logger.info(f"Loaded distances from cache: {len(distances_df)} pairs")
            return distances_df
        except Exception:
            pass
    distances_df = calculate_hospital_distances(cms_df, max_distance=max_distance)
    if distances_df is None or len(distances_df) == 0:
        return distances_df
    distances_df = distances_df[distances_df['Distance_Miles'] >= min_distance].reset_index(drop=True)
    try:
        with open(cache_pkl, 'wb') as f:
            pickle.dump(distances_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cached distances: {len(distances_df)} pairs")
    except Exception as e:
        logger.warning(f"Failed to cache distances: {e}")
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(os.path.dirname(__file__), f'hospital_distances_<=35mi_{ts}.csv')
        distances_df.to_csv(csv_path, index=False)
        logger.info(f"Saved distances CSV: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save distances CSV: {e}")
    return distances_df

def compute_potential_cah(cms_df, distances_df, beds_limit=25):
    """Compute Potential_CAH using distance results plus CAH exclusion and bed limit."""
    try:
        cms_df = identify_potential_cah(cms_df, distances_df=distances_df, min_distance=1.0, max_distance=35.0)
    except Exception as e:
        logger.warning(f'identify_potential_cah failed, defaulting to zeros: {e}')
        cms_df['Potential_CAH'] = 0
        return cms_df
    if 'Facility Type' in cms_df.columns:
        cms_df.loc[cms_df['Facility Type'] == 'Critical Access Hospital', 'Potential_CAH'] = 0
    if 'Number Beds' in cms_df.columns:
        nb = pd.to_numeric(cms_df['Number Beds'], errors='coerce')
        cms_df.loc[nb > beds_limit, 'Potential_CAH'] = 0
    cms_df['Potential_CAH'] = cms_df['Potential_CAH'].fillna(0).astype(int)
    return cms_df

def tag_financial_heuristics(cms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add financial heuristic tags to cms_df based on rules provided by the user.
    This function is robust to missing years and missing numeric values.
    """
    df = cms_df.copy()

    # Coerce relevant numeric columns
    num_cols = [
        'EBITDAR', 'Net Income', 'Total Current Liabilities', 'Total Long-Term Liabilities',
        'Cash', 'Operating Expense', 'Other Expense', 'Total Current Assets',
        'Government Reimbursements', 'Net Receivables'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # >$400k EBITDAR
    if 'EBITDAR' in df.columns:
        df['>$400k EBITDAR'] = (df['EBITDAR'] > 400000).astype(int)

    # Debt <$5M or < 5x Net Income (average by Hospital Name)
    if {'Total Current Liabilities', 'Total Long-Term Liabilities', 'Net Income', 'Hospital Name'}.issubset(df.columns):
        df['Net Income Average'] = df.groupby('Hospital Name')['Net Income'].transform('mean')
        df['Total Debt'] = df[['Total Current Liabilities', 'Total Long-Term Liabilities']].sum(axis=1, min_count=1)
        df['Debt <$5M or 5x NI'] = (
            (df['Total Debt'] < 5_000_000) |
            (df['Total Debt'] < 5 * df['Net Income Average'])
        ).astype(int)

    # Newly Positive after two negative NI years (by Hospital Name)
    if {'Hospital Name', 'Year', 'Net Income'}.issubset(df.columns):
        ni_pivot = df.pivot_table(index='Hospital Name', columns='Year', values='Net Income', aggfunc='mean')
        years = sorted([y for y in ni_pivot.columns if pd.notna(y)])

        def compute_newly_positive(row):
            # Find most recent year with data
            recent_year = None
            for y in reversed(years):
                v = row.get(y, np.nan)
                if pd.notna(v):
                    recent_year = y
                    break
            if recent_year is None:
                return 0
            y0 = recent_year
            y1 = years[years.index(y0)-1] if y0 in years and years.index(y0) - 1 >= 0 else None
            y2 = years[years.index(y0)-2] if y0 in years and years.index(y0) - 2 >= 0 else None

            now_pos = 1 if pd.notna(row.get(y0, np.nan)) and row.get(y0, 0) > 0 else 0
            prev_neg = 0
            if y1 is not None and y2 is not None:
                v1 = row.get(y1, np.nan)
                v2 = row.get(y2, np.nan)
                if pd.notna(v1) and pd.notna(v2) and (v1 < 0) and (v2 < 0):
                    prev_neg = 1
            return 1 if (now_pos == 1 and prev_neg == 1) else 0

        newly_pos_series = ni_pivot.apply(compute_newly_positive, axis=1)
        newly_pos = newly_pos_series.reset_index()
        newly_pos.columns = ['Hospital Name', 'Newly Positive']
        df = df.merge(newly_pos, on='Hospital Name', how='left')
        df['Newly Positive'] = df['Newly Positive'].fillna(0).astype(int)

    # Positive cash position
    if 'Cash' in df.columns:
        df['Positive Cash Position'] = (df['Cash'] > 0).astype(int)

    # Cash vs 65 days
    if {'Cash', 'Operating Expense'}.issubset(df.columns):
        df['Cash vs. 65 Days'] = (df['Cash'] > (df['Operating Expense'] / 365.0 * 65.0)).astype(int)

    # Total expenses + Income vs 3% of total expenses
    if {'Operating Expense', 'Other Expense', 'Net Income'}.issubset(df.columns):
        df['Total Expenses'] = df[['Operating Expense', 'Other Expense']].sum(axis=1, min_count=1)
        df['Income vs. 3% Total Expenses'] = (df['Net Income'] > (0.03 * df['Total Expenses'])).astype(int)

    # Assets vs 100 days
    if {'Total Current Assets', 'Operating Expense'}.issubset(df.columns):
        df['Assets vs. 100 Days'] = (df['Total Current Assets'] > (df['Operating Expense'] / 365.0 * 100.0)).astype(int)

    # $100k Reimbursement
    if 'Government Reimbursements' in df.columns:
        df['$100k Reimbursement'] = (df['Government Reimbursements'] > 100000).astype(int)

    # Net Receivables vs 88 Days (implement as code provided: tag when greater than 88 days)
    if {'Net Receivables', 'Operating Expense'}.issubset(df.columns):
        df['Net Receivables vs. 88 Days'] = (df['Net Receivables'] > (df['Operating Expense'] / 365.0 * 88.0)).astype(int)

    # Debt change indicators: Just Built and Ready to Build (by CCN)
    if {'CCN', 'Year'}.issubset(df.columns):
        # Pivots
        debt_piv = df.pivot_table(index='CCN', columns='Year', values='Total Long-Term Liabilities', aggfunc='mean')
        ebitdar_piv = df.pivot_table(index='CCN', columns='Year', values='EBITDAR', aggfunc='mean')
        years_ccn = sorted([y for y in set(debt_piv.columns.tolist()) | set(ebitdar_piv.columns.tolist())])

        # Compute deltas over last up to 3 intervals
        def compute_debt_flags(row):
            # Just Built: any of last up to 3 deltas > 0.25
            deltas = []
            for i in range(1, len(years_ccn)):
                y_prev = years_ccn[i-1]
                y_curr = years_ccn[i]
                prev = row.get(y_prev, np.nan)
                curr = row.get(y_curr, np.nan)
                if pd.notna(prev) and prev != 0 and pd.notna(curr):
                    deltas.append((curr - prev) / prev)
            last_three = deltas[-3:] if len(deltas) >= 3 else deltas
            just_built = 1 if any(d > 0.25 for d in last_three if pd.notna(d)) else 0
            return pd.Series({'Just Built': just_built})

        just_built_df = debt_piv.apply(compute_debt_flags, axis=1)

        # Ready to Build: latest liabilities / latest ebitdar between 0 and 0.25
        def compute_ready_to_build(ccn):
            liab_row = debt_piv.loc[ccn] if ccn in debt_piv.index else None
            e_row = ebitdar_piv.loc[ccn] if ccn in ebitdar_piv.index else None
            if liab_row is None or e_row is None:
                return 0
            # Find the most recent year where both are present
            for y in reversed(years_ccn):
                lv = liab_row.get(y, np.nan)
                ev = e_row.get(y, np.nan)
                if pd.notna(lv) and pd.notna(ev) and ev != 0:
                    ratio = lv / ev
                    return 1 if (ratio < 0.25 and ratio > 0) else 0
            return 0

        ready_to_build_series = pd.Series({ccn: compute_ready_to_build(ccn) for ccn in debt_piv.index})
        ready_to_build_df = ready_to_build_series.to_frame(name='Ready to Build')

        flags = just_built_df.join(ready_to_build_df, how='outer').reset_index()
        df = df.merge(flags[['CCN', 'Just Built', 'Ready to Build']], on='CCN', how='left')
        df[['Just Built', 'Ready to Build']] = df[['Just Built', 'Ready to Build']].fillna(0).astype(int)

    return df

# ------------------------- ML: Build_Odds Pipeline -------------------------
def _ml_build_feature_matrix(cms_df: pd.DataFrame, target_col: str = 'Ready to Build'):
    """Create feature matrix X and target y from cms_df using numeric columns only.
    Falls back to 'Just Built' if target_col not present. Returns (X, y, feature_cols).
    """
    tgt = target_col if target_col in cms_df.columns else ('Just Built' if 'Just Built' in cms_df.columns else None)
    if tgt is None:
        raise ValueError("No target column ('Ready to Build' or 'Just Built') found to train Build_Odds model")
    # Select numeric features
    num_df = cms_df.select_dtypes(include=[np.number]).copy()
    # Drop target from features if present
    if tgt in num_df.columns:
        num_df = num_df.drop(columns=[tgt])
    # Basic cleaning
    num_df = num_df.fillna(0)
    # Align target
    y = cms_df[tgt].astype(int)
    X = num_df
    feature_cols = X.columns.tolist()
    # Ensure we have at least a few features
    if X.shape[1] == 0:
        raise ValueError("No numeric features available to train Build_Odds model")
    return X, y, feature_cols

def _train_build_odds_model(X: pd.DataFrame, y: pd.Series):
    """Train an XGBoost classifier. If only one class present, return None and baseline prob."""
    try:
        if y.nunique() < 2:
            baseline = float(y.mean()) if len(y) > 0 else 0.0
            return None, baseline
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=4,
        )
        model.fit(X, y)
        return model, None
    except Exception as e:
        logger.warning(f"Training Build_Odds model failed: {e}")
        baseline = float(y.mean()) if len(y) > 0 else 0.0
        return None, baseline

def add_build_odds(cms_df: pd.DataFrame, target_col: str = 'Ready to Build') -> pd.DataFrame:
    """Train on heuristic target and add 'Build_Odds' probability column for all rows."""
    try:
        X, y, feature_cols = _ml_build_feature_matrix(cms_df, target_col=target_col)
        model, baseline = _train_build_odds_model(X, y)
        if model is None:
            probs = np.full(shape=(len(cms_df),), fill_value=baseline, dtype=float)
        else:
            # Build features for all rows using same columns
            full_num = cms_df.select_dtypes(include=[np.number]).copy().fillna(0)
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in full_num.columns:
                    full_num[col] = 0
            full_X = full_num[feature_cols]
            probs = model.predict_proba(full_X)[:, 1]
        cms_df = cms_df.copy()
        cms_df['Build_Odds'] = probs
        return cms_df
    except Exception as e:
        logger.warning(f"add_build_odds failed: {e}")
        cms_df = cms_df.copy()
        cms_df['Build_Odds'] = np.nan
        return cms_df

def extract_additional_financial_metrics(numeric_data, use_cache=True):
    """
    Extract additional financial metrics from numeric dataset including liabilities,
    assets, operating costs, receivables, cash, and provider types.
    
    Args:
        numeric_data (pandas.DataFrame): The numeric data to extract financial metrics from
        use_cache (bool): If True, use cached data if available
        
    Returns:
        dict: Dictionary containing DataFrames for various financial metrics
    """
    global _liabilities_df, _longLiabilities_df, _operatingCosts_df, _otherCosts_df
    global _assets_df, _reimburse_df, _receivables_df, _cash_df, _provider_df
    
    # Check if all data is already loaded in memory
    if use_cache and all(df is not None for df in [
        _liabilities_df, _longLiabilities_df, _operatingCosts_df, _otherCosts_df,
        _assets_df, _reimburse_df, _receivables_df, _cash_df, _provider_df
    ]):
        logger.info("Using in-memory cached additional financial metrics")
        return {
            'liabilities': _liabilities_df,
            'long_liabilities': _longLiabilities_df,
            'operating_costs': _operatingCosts_df,
            'other_costs': _otherCosts_df,
            'assets': _assets_df,
            'reimbursements': _reimburse_df,
            'receivables': _receivables_df,
            'cash': _cash_df,
            'provider_types': _provider_df
        }
    
    # Check if cached data exists on disk
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    metrics = {}
    cache_files = {
        'liabilities': os.path.join(cache_dir, 'liabilities_data.pkl'),
        'long_liabilities': os.path.join(cache_dir, 'long_liabilities_data.pkl'),
        'operating_costs': os.path.join(cache_dir, 'operating_costs_data.pkl'),
        'other_costs': os.path.join(cache_dir, 'other_costs_data.pkl'),
        'assets': os.path.join(cache_dir, 'assets_data.pkl'),
        'reimbursements': os.path.join(cache_dir, 'reimbursements_data.pkl'),
        'receivables': os.path.join(cache_dir, 'receivables_data.pkl'),
        'cash': os.path.join(cache_dir, 'cash_data.pkl'),
        'provider_types': os.path.join(cache_dir, 'provider_types_data.pkl')
    }
    
    # Try to load from disk cache first
    if use_cache:
        all_loaded = True
        for metric_name, cache_path in cache_files.items():
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        metrics[metric_name] = pickle.load(f)
                        logger.info(f"Loaded {len(metrics[metric_name])} {metric_name} records from disk cache")
                except Exception as e:
                    logger.warning(f"Failed to load {metric_name} cache: {str(e)}")
                    all_loaded = False
            else:
                all_loaded = False
        
        if all_loaded:
            # Update global variables
            _liabilities_df = metrics['liabilities']
            _longLiabilities_df = metrics['long_liabilities']
            _operatingCosts_df = metrics['operating_costs']
            _otherCosts_df = metrics['other_costs']
            _assets_df = metrics['assets']
            _reimburse_df = metrics['reimbursements']
            _receivables_df = metrics['receivables']
            _cash_df = metrics['cash']
            _provider_df = metrics['provider_types']
            
            logger.info("Successfully loaded all additional financial metrics from cache")
            return metrics
    
    logger.info("Extracting additional financial metrics from source data")
    
    try:
        # Get total current liabilities
        _liabilities_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                         (numeric_data['Column Number'].astype(str) == '00100') & 
                                         (numeric_data['Line Number'] == 4500)].copy()
        _liabilities_df = _liabilities_df.rename(columns={'Value': 'Total Current Liabilities'})
        _liabilities_df.reset_index(drop=True, inplace=True)
        metrics['liabilities'] = _liabilities_df
        logger.info(f"Extracted {len(_liabilities_df)} current liabilities records")
        
        # Get total long-term liabilities
        _longLiabilities_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                             (numeric_data['Column Number'].astype(str) == '00100') & 
                                             (numeric_data['Line Number'] == 5000)].copy()
        _longLiabilities_df = _longLiabilities_df.rename(columns={'Value': 'Total Long-Term Liabilities'})
        _longLiabilities_df.reset_index(drop=True, inplace=True)
        metrics['long_liabilities'] = _longLiabilities_df
        logger.info(f"Extracted {len(_longLiabilities_df)} long-term liabilities records")
        
        # Get total operating expense
        _operatingCosts_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G300000') & 
                                            (numeric_data['Column Number'].astype(str) == '00100') & 
                                            (numeric_data['Line Number'] == 400)].copy()
        _operatingCosts_df = _operatingCosts_df.rename(columns={'Value': 'Operating Expense'})
        _operatingCosts_df.reset_index(drop=True, inplace=True)
        metrics['operating_costs'] = _operatingCosts_df
        logger.info(f"Extracted {len(_operatingCosts_df)} operating expense records")
        
        # Get other expenses
        _otherCosts_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G300000') & 
                                        (numeric_data['Column Number'].astype(str) == '00100') & 
                                        (numeric_data['Line Number'] == 2800)].copy()
        _otherCosts_df = _otherCosts_df.rename(columns={'Value': 'Other Expense'})
        _otherCosts_df.reset_index(drop=True, inplace=True)
        metrics['other_costs'] = _otherCosts_df
        logger.info(f"Extracted {len(_otherCosts_df)} other expense records")
        
        # Get total current assets
        _assets_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                    (numeric_data['Column Number'].astype(str) == '00100') & 
                                    (numeric_data['Line Number'] == 1100)].copy()
        _assets_df = _assets_df.rename(columns={'Value': 'Total Current Assets'})
        _assets_df.reset_index(drop=True, inplace=True)
        metrics['assets'] = _assets_df
        logger.info(f"Extracted {len(_assets_df)} current assets records")
        
        # Get government reimbursements
        _reimburse_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S100000') & 
                                       (numeric_data['Column Number'].astype(str) == '00100') & 
                                       (numeric_data['Line Number'] == 1800)].copy()
        _reimburse_df = _reimburse_df.rename(columns={'Value': 'Government Reimbursements'})
        _reimburse_df.reset_index(drop=True, inplace=True)
        metrics['reimbursements'] = _reimburse_df
        logger.info(f"Extracted {len(_reimburse_df)} government reimbursement records")
        
        # Get net receivables components
        notes_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                   (numeric_data['Column Number'].astype(str) == '00100') & 
                                   (numeric_data['Line Number'] == 300)].copy()
        accounts_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                      (numeric_data['Column Number'].astype(str) == '00100') & 
                                      (numeric_data['Line Number'] == 400)].copy()
        other_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                   (numeric_data['Column Number'].astype(str) == '00100') & 
                                   (numeric_data['Line Number'] == 500)].copy()
        allowances_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                        (numeric_data['Column Number'].astype(str) == '00100') & 
                                        (numeric_data['Line Number'] == 600)].copy()
        
        notes_df = notes_df.rename(columns={'Value': 'Notes'})
        accounts_df = accounts_df.rename(columns={'Value': 'Accounts'})
        other_df = other_df.rename(columns={'Value': 'Other'})
        allowances_df = allowances_df.rename(columns={'Value': 'Allowances'})
        
        # Combine receivables components
        _receivables_df = notes_df.merge(accounts_df[['Report Record Number', 'Year', 'Accounts']], 
                                       on=['Report Record Number', 'Year'], how='outer')
        _receivables_df = _receivables_df.merge(other_df[['Report Record Number', 'Year', 'Other']], 
                                            on=['Report Record Number', 'Year'], how='outer')
        _receivables_df = _receivables_df.merge(allowances_df[['Report Record Number', 'Year', 'Allowances']], 
                                             on=['Report Record Number', 'Year'], how='outer')
        
        _receivables_df['Notes'] = _receivables_df['Notes'].fillna(0)
        _receivables_df['Accounts'] = _receivables_df['Accounts'].fillna(0)
        _receivables_df['Other'] = _receivables_df['Other'].fillna(0)
        _receivables_df['Allowances'] = _receivables_df['Allowances'].fillna(0)
        
        _receivables_df['Net Receivables'] = _receivables_df['Notes'] + _receivables_df['Accounts'] + _receivables_df['Other'] - _receivables_df['Allowances']
        metrics['receivables'] = _receivables_df
        logger.info(f"Calculated net receivables for {len(_receivables_df)} records")
        
        # Get cash on hand
        _cash_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & 
                                  (numeric_data['Column Number'].astype(str) == '00100') & 
                                  (numeric_data['Line Number'] == 100)].copy()
        _cash_df = _cash_df.rename(columns={'Value': 'Cash'})
        _cash_df.reset_index(drop=True, inplace=True)
        metrics['cash'] = _cash_df
        logger.info(f"Extracted {len(_cash_df)} cash records")
        
        # Get provider types
        _provider_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S200001') & 
                                      (numeric_data['Column Number'].astype(str) == '00400') & 
                                      (numeric_data['Line Number'] == 300)].copy()
        _provider_df = _provider_df.rename(columns={'Value': 'Provider Type'})
        _provider_df.reset_index(drop=True, inplace=True)
        metrics['provider_types'] = _provider_df
        logger.info(f"Extracted {len(_provider_df)} provider type records")
        
        # Save all metrics to disk cache
        if use_cache:
            for metric_name, df in metrics.items():
                with open(cache_files[metric_name], 'wb') as f:
                    pickle.dump(df, f)
                    logger.info(f"Saved {len(df)} {metric_name} records to disk cache")
        
        logger.info("Successfully extracted all additional financial metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error extracting additional financial metrics: {str(e)}")
        raise

def analyze_financial_metrics(ebitda_df, use_cache=True):
    """
    Analyze financial metrics from EBITDA data.
    
    Args:
        ebitda_df (pandas.DataFrame): DataFrame containing EBITDA calculations
        use_cache (bool): If True, use cached analysis if available
        
    Returns:
        dict: Dictionary containing financial metrics and analysis
    """
    logger.info("Analyzing financial metrics")
    
    try:
        # Calculate basic statistics
        metrics = {}
        
        # Basic EBITDA statistics
        metrics['ebitda_mean'] = ebitda_df['EBITDA'].mean()
        metrics['ebitda_median'] = ebitda_df['EBITDA'].median()
        metrics['ebitda_std'] = ebitda_df['EBITDA'].std()
        metrics['ebitda_min'] = ebitda_df['EBITDA'].min()
        metrics['ebitda_max'] = ebitda_df['EBITDA'].max()
        metrics['ebitda_total'] = ebitda_df['EBITDA'].sum()
        
        # EBITDAR statistics if available
        if 'EBITDAR' in ebitda_df.columns:
            metrics['ebitdar_mean'] = ebitda_df['EBITDAR'].mean()
            metrics['ebitdar_median'] = ebitda_df['EBITDAR'].median()
            metrics['ebitdar_std'] = ebitda_df['EBITDAR'].std()
            metrics['ebitdar_min'] = ebitda_df['EBITDAR'].min()
            metrics['ebitdar_max'] = ebitda_df['EBITDAR'].max()
            metrics['ebitdar_total'] = ebitda_df['EBITDAR'].sum()
        
        # Calculate EBITDA by year
        ebitda_by_year = {}
        for year, group in ebitda_df.groupby('Year'):
            ebitda_by_year[year] = {
                'mean': group['EBITDA'].mean(),
                'median': group['EBITDA'].median(),
                'sum': group['EBITDA'].sum(),
                'count': len(group)
            }
        metrics['ebitda_by_year'] = ebitda_by_year
        
        # Calculate year-over-year growth rates
        years = sorted(ebitda_by_year.keys())
        ebitda_yoy_growth = {}
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            prev_sum = ebitda_by_year[prev_year]['sum']
            curr_sum = ebitda_by_year[curr_year]['sum']
            
            if prev_sum != 0:  # Avoid division by zero
                growth_rate = ((curr_sum - prev_sum) / abs(prev_sum)) * 100
                ebitda_yoy_growth[f"{prev_year}-{curr_year}"] = growth_rate
        
        metrics['ebitda_yoy_growth'] = ebitda_yoy_growth
        
        logger.info("Financial metrics analysis complete")
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing financial metrics: {str(e)}")
        raise

def analyze_numeric_data(numeric_data, alpha_data=None, use_cache=True):
    """
    Analyze the numeric data to extract useful insights, including financial metrics and EBITDA analysis.
    
    Args:
        numeric_data (pandas.DataFrame): The numeric data to analyze
        alpha_data (pandas.DataFrame, optional): The alpha data containing hospital information
        use_cache (bool): If True, use cached data if available. If False, always extract from source data.
    
    Returns:
        dict: Dictionary containing various analysis results including financial metrics
    """
    logger.info("Analyzing numeric data")
    
    # Initialize results dictionary
    results = {}
    
    try:
        # Basic statistics
        results['total_records'] = len(numeric_data)
        results['years_available'] = sorted(numeric_data['Year'].unique().tolist())
        results['unique_report_records'] = numeric_data['Report Record Number'].nunique()
        results['unique_worksheets'] = numeric_data['Worksheet Code'].nunique()
        results['unique_form_ids'] = numeric_data['Form ID'].nunique()
        
        # Value statistics (convert to numeric first, ignoring errors)
        numeric_data['Value_numeric'] = pd.to_numeric(numeric_data['Value'], errors='coerce')
        results['value_mean'] = numeric_data['Value_numeric'].mean()
        results['value_median'] = numeric_data['Value_numeric'].median()
        results['value_min'] = numeric_data['Value_numeric'].min()
        results['value_max'] = numeric_data['Value_numeric'].max()
        results['value_null_count'] = numeric_data['Value_numeric'].isna().sum()
        
        # Most common worksheets
        worksheet_counts = numeric_data['Worksheet Code'].value_counts().head(10).to_dict()
        results['top_worksheets'] = worksheet_counts
        
        # Year-by-year record counts
        year_counts = numeric_data.groupby('Year').size().to_dict()
        results['records_by_year'] = year_counts
        
        # Extract financial data and calculate EBITDA
        logger.info("Extracting financial data for EBITDA analysis")
        netIncome_df, interest_df, taxes_df, depreciation_df, rent_df = extract_financial_data(numeric_data, use_cache=use_cache)
        
        # Calculate EBITDA
        logger.info("Calculating EBITDA metrics")
        ebitda_df = calculate_ebitda(netIncome_df, interest_df, taxes_df, depreciation_df, rent_df, use_cache=use_cache)
        
        # Extract additional financial metrics
        logger.info("Extracting additional financial metrics")
        additional_metrics = extract_additional_financial_metrics(numeric_data, use_cache=use_cache)
        
        # Analyze financial metrics
        logger.info("Analyzing financial metrics")
        financial_results = analyze_financial_metrics(ebitda_df)
        
        # Add financial results to the main results dictionary
        results['financial_metrics'] = financial_results
        
        # Add additional financial metrics to results
        results['additional_financial_metrics'] = {
            'liabilities': {
                'count': len(additional_metrics['liabilities']),
                'mean': additional_metrics['liabilities']['Total Current Liabilities'].mean() if 'liabilities' in additional_metrics else None,
                'median': additional_metrics['liabilities']['Total Current Liabilities'].median() if 'liabilities' in additional_metrics else None,
                'sum': additional_metrics['liabilities']['Total Current Liabilities'].sum() if 'liabilities' in additional_metrics else None
            },
            'long_term_liabilities': {
                'count': len(additional_metrics['long_liabilities']),
                'mean': additional_metrics['long_liabilities']['Total Long-Term Liabilities'].mean() if 'long_liabilities' in additional_metrics else None,
                'median': additional_metrics['long_liabilities']['Total Long-Term Liabilities'].median() if 'long_liabilities' in additional_metrics else None,
                'sum': additional_metrics['long_liabilities']['Total Long-Term Liabilities'].sum() if 'long_liabilities' in additional_metrics else None
            },
            'operating_costs': {
                'count': len(additional_metrics['operating_costs']),
                'mean': additional_metrics['operating_costs']['Operating Expense'].mean() if 'operating_costs' in additional_metrics else None,
                'median': additional_metrics['operating_costs']['Operating Expense'].median() if 'operating_costs' in additional_metrics else None,
                'sum': additional_metrics['operating_costs']['Operating Expense'].sum() if 'operating_costs' in additional_metrics else None
            },
            'other_costs': {
                'count': len(additional_metrics['other_costs']),
                'mean': additional_metrics['other_costs']['Other Expense'].mean() if 'other_costs' in additional_metrics else None,
                'median': additional_metrics['other_costs']['Other Expense'].median() if 'other_costs' in additional_metrics else None,
                'sum': additional_metrics['other_costs']['Other Expense'].sum() if 'other_costs' in additional_metrics else None
            },
            'assets': {
                'count': len(additional_metrics['assets']),
                'mean': additional_metrics['assets']['Total Current Assets'].mean() if 'assets' in additional_metrics else None,
                'median': additional_metrics['assets']['Total Current Assets'].median() if 'assets' in additional_metrics else None,
                'sum': additional_metrics['assets']['Total Current Assets'].sum() if 'assets' in additional_metrics else None
            },
            'reimbursements': {
                'count': len(additional_metrics['reimbursements']),
                'mean': additional_metrics['reimbursements']['Government Reimbursements'].mean() if 'reimbursements' in additional_metrics else None,
                'median': additional_metrics['reimbursements']['Government Reimbursements'].median() if 'reimbursements' in additional_metrics else None,
                'sum': additional_metrics['reimbursements']['Government Reimbursements'].sum() if 'reimbursements' in additional_metrics else None
            },
            'receivables': {
                'count': len(additional_metrics['receivables']),
                'mean': additional_metrics['receivables']['Net Receivables'].mean() if 'receivables' in additional_metrics else None,
                'median': additional_metrics['receivables']['Net Receivables'].median() if 'receivables' in additional_metrics else None,
                'sum': additional_metrics['receivables']['Net Receivables'].sum() if 'receivables' in additional_metrics else None
            },
            'cash': {
                'count': len(additional_metrics['cash']),
                'mean': additional_metrics['cash']['Cash'].mean() if 'cash' in additional_metrics else None,
                'median': additional_metrics['cash']['Cash'].median() if 'cash' in additional_metrics else None,
                'sum': additional_metrics['cash']['Cash'].sum() if 'cash' in additional_metrics else None
            },
            'provider_types': {
                'count': len(additional_metrics['provider_types']),
                'unique_types': additional_metrics['provider_types']['Provider Type'].nunique() if 'provider_types' in additional_metrics else None,
                'type_distribution': additional_metrics['provider_types']['Provider Type'].value_counts().to_dict() if 'provider_types' in additional_metrics else None
            }
        }
        
        # Add summary of financial data to results
        results['financial_summary'] = {
            'net_income_records': len(netIncome_df) if netIncome_df is not None else 0,
            'interest_records': len(interest_df) if interest_df is not None else 0,
            'taxes_records': len(taxes_df) if taxes_df is not None else 0,
            'depreciation_records': len(depreciation_df) if depreciation_df is not None else 0,
            'rent_records': len(rent_df) if rent_df is not None else 0,
            'ebitda_records': len(ebitda_df) if ebitda_df is not None else 0,
            'liabilities_records': len(additional_metrics['liabilities']) if 'liabilities' in additional_metrics else 0,
            'long_term_liabilities_records': len(additional_metrics['long_liabilities']) if 'long_liabilities' in additional_metrics else 0,
            'operating_costs_records': len(additional_metrics['operating_costs']) if 'operating_costs' in additional_metrics else 0,
            'other_costs_records': len(additional_metrics['other_costs']) if 'other_costs' in additional_metrics else 0,
            'assets_records': len(additional_metrics['assets']) if 'assets' in additional_metrics else 0,
            'reimbursements_records': len(additional_metrics['reimbursements']) if 'reimbursements' in additional_metrics else 0,
            'receivables_records': len(additional_metrics['receivables']) if 'receivables' in additional_metrics else 0,
            'cash_records': len(additional_metrics['cash']) if 'cash' in additional_metrics else 0,
            'provider_types_records': len(additional_metrics['provider_types']) if 'provider_types' in additional_metrics else 0
        }
        
        # Log some key findings
        logger.info(f"Numeric data analysis complete - {results['total_records']:,} total records across {len(results['years_available'])} years")
        logger.info(f"Found {results['unique_report_records']:,} unique report records")
        logger.info(f"Found {results['unique_worksheets']:,} unique worksheet codes")
        logger.info(f"Found {results['unique_form_ids']:,} unique form IDs")
        logger.info(f"EBITDA analysis complete - {results['financial_summary']['ebitda_records']:,} EBITDA records calculated")
        logger.info(f"Additional financial metrics extracted - {results['financial_summary']['liabilities_records']:,} liabilities records, {results['financial_summary']['assets_records']:,} assets records")
        
        # Create comprehensive hospital dataset by merging all metrics
        logger.info("Creating comprehensive hospital dataset by merging all metrics")
        
        # Extract hospital information from alpha data
        logger.info("Using hospital information from alpha data")
        date_df, names_df, zip_df, address_df, city_df, states_df, ccn_df, control_df, network_df = extract_hospital_data(alpha_data, use_cache=use_cache)
        
        # Extract hospital metrics from numeric data
        logger.info("Using hospital metrics from numeric data")
        beds_df, discharge_df, rural_df, charity_df, uncompensated_df = extract_hospital_metrics(numeric_data, use_cache=use_cache)
        
        # Get the additional financial metrics DataFrames
        liabilities_df = additional_metrics['liabilities']
        longLiabilities_df = additional_metrics['long_liabilities']
        operatingCosts_df = additional_metrics['operating_costs']
        otherCosts_df = additional_metrics['other_costs']
        assets_df = additional_metrics['assets']
        reimburse_df = additional_metrics['reimbursements']
        receivables_df = additional_metrics['receivables']
        cash_df = additional_metrics['cash']
        provider_df = additional_metrics['provider_types']
        
        # Merge all hospital metrics and financial data into a single DataFrame
        try:
            logger.info("Starting hospital data merging process")
            
            # Start with names_df as the base
            if 'Report Record Number' not in names_df.columns or 'Hospital Name' not in names_df.columns or 'Year' not in names_df.columns:
                logger.error(f"Missing required columns in names_df. Columns: {names_df.columns.tolist()}")
                raise ValueError("Missing required columns in names_df")
                
            cms_df = names_df[['Report Record Number', 'Hospital Name', 'Year']].copy()
            logger.info(f"Base DataFrame created with {len(cms_df)} records")
            
            # Helper function to safely merge DataFrames
            def safe_merge(left_df, right_df, right_columns, on_columns, how='outer'):
                if not all(col in right_df.columns for col in right_columns):
                    logger.warning(f"Missing columns in right DataFrame. Expected {right_columns}, got {right_df.columns.tolist()}")
                    return left_df
                if not all(col in left_df.columns for col in on_columns):
                    logger.warning(f"Missing join columns in left DataFrame. Expected {on_columns}, got {left_df.columns.tolist()}")
                    return left_df
                
                try:
                    result = left_df.merge(right_df[right_columns], on=on_columns, how=how)
                    logger.info(f"Successfully merged {right_columns[1:]} with {len(result)} resulting records")
                    return result
                except Exception as e:
                    logger.error(f"Error merging {right_columns[1:]}: {str(e)}")
                    return left_df
            
            # Perform all merges using the safe_merge function
            cms_df = safe_merge(cms_df, states_df, ['Report Record Number', 'State'], ['Report Record Number'])
            cms_df = safe_merge(cms_df, city_df, ['Report Record Number', 'City', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, address_df, ['Report Record Number', 'Address', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, zip_df, ['Report Record Number', 'Zip Code', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, date_df, ['Report Record Number', 'Report Start Date', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, ccn_df, ['Report Record Number', 'CCN', 'Facility Type', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, control_df, ['Report Record Number', 'Type of Control', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, network_df, ['Report Record Number', 'Network', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, beds_df, ['Report Record Number', 'Number Beds', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, discharge_df, ['Report Record Number', 'Discharges', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, rural_df, ['Report Record Number', 'Rural vs. Urban', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, charity_df, ['Report Record Number', 'Cost of Charity', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, uncompensated_df, ['Report Record Number', 'Cost of Uncompensated Care', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, netIncome_df, ['Report Record Number', 'Net Income', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, ebitda_df, ['Report Record Number', 'EBITDAR', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, liabilities_df, ['Report Record Number', 'Total Current Liabilities', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, longLiabilities_df, ['Report Record Number', 'Total Long-Term Liabilities', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, cash_df, ['Report Record Number', 'Cash', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, operatingCosts_df, ['Report Record Number', 'Operating Expense', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, otherCosts_df, ['Report Record Number', 'Other Expense', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, reimburse_df, ['Report Record Number', 'Government Reimbursements', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, receivables_df, ['Report Record Number', 'Net Receivables', 'Year'], ['Report Record Number', 'Year'])
            cms_df = safe_merge(cms_df, assets_df, ['Report Record Number', 'Total Current Assets', 'Year'], ['Report Record Number', 'Year'])
            
            logger.info(f"All merges completed successfully with final shape: {cms_df.shape}")
        except Exception as e:
            logger.error(f"Error during hospital data merging: {str(e)}")
            raise

        # Mark independent hospitals
        cms_df['Check'] = cms_df['Network'].fillna('Independent')

        # Clean zip codes
        cms_df['Zip Code'] = cms_df['Zip Code'].str.split('-').str[0]
        
        # Create Full Address column for geolocation
        logger.info("Creating Full Address column for geolocation")
        # Ensure all address components are strings before concatenation
        cms_df['Address'] = cms_df['Address'].astype(str)
        cms_df['City'] = cms_df['City'].astype(str)
        cms_df['State'] = cms_df['State'].astype(str)
        cms_df['Zip Code'] = cms_df['Zip Code'].astype(str)
        
        # Concatenate address components into a full address
        cms_df['Full Address'] = cms_df['Address'] + ', ' + cms_df['City'] + ', ' + cms_df['State'] + ', ' + cms_df['Zip Code']
        logger.info(f"Created Full Address column for {len(cms_df)} records")

        # Total hospitals
        hospitals = len(cms_df['Hospital Name'].unique())
        logger.info(f"Created comprehensive dataset with {len(cms_df)} records from {hospitals} unique hospitals")
        
        # Add comprehensive hospital data to results
        results['comprehensive_hospital_data'] = cms_df
        results['total_hospitals'] = hospitals
        
        # Convert latitude and longitude to radians if they exist
        if 'Latitude' in cms_df.columns and 'Longitude' in cms_df.columns:
            # Convert lat / longs from object to float 64
            cms_df['Latitude'] = cms_df['Latitude'].astype('float64')
            cms_df['Longitude'] = cms_df['Longitude'].astype('float64')
            
            # Define function to convert degrees to radians
            def deg_to_rad(angle):
                return (angle * math.pi) / 180
            
            # Convert to radians
            cms_df['Lat'] = deg_to_rad(cms_df['Latitude'])
            cms_df['Long'] = deg_to_rad(cms_df['Longitude'])
            
            logger.info(f"Converted {cms_df['Latitude'].notna().sum()} coordinates from degrees to radians")
            
            # Update the results with the converted coordinates
            results['comprehensive_hospital_data'] = cms_df
            
            # Identify potential Critical Access Hospitals
            logger.info("Identifying potential Critical Access Hospitals")
            cms_df = identify_potential_cah(cms_df)
            
            # Update the results with CAH information
            results['comprehensive_hospital_data'] = cms_df
            results['potential_cah_count'] = cms_df['Potential_CAH'].sum()
        
    except Exception as e:
        logger.error(f"Error analyzing numeric data: {str(e)}")
        results['error'] = str(e)
    
    return results

if __name__ == "__main__":
    # By default, use caching for local development
    # Set use_cache=False to always load fresh data (useful for Azure Function deployment)
    alpha_data, numeric_data, beds_df, discharge_df, rural_df, charity_df, uncompensated_df = main(use_cache=True)
    
    # Print basic information about the numeric data
    print(f"\nNumeric Data Information:")
    print(f"Shape: {numeric_data.shape}")
    print(f"Columns: {numeric_data.columns.tolist()}")
    print(f"Years: {sorted(numeric_data['Year'].unique())}")
    
    # Print information about the extracted hospital metrics
    print(f"\nExtracted Hospital Metrics:")
    print(f"Number of hospitals with bed data: {len(beds_df)}")
    print(f"Number of hospitals with discharge data: {len(discharge_df)}")
    print(f"Number of hospitals with rural/urban status: {len(rural_df)}")
    print(f"Number of hospitals with charity cost data: {len(charity_df)}")
    print(f"Number of hospitals with uncompensated care data: {len(uncompensated_df)}")
    
    # Display samples of the extracted metrics
    print(f"\nSample of beds data:")
    print(beds_df.head())
    
    print(f"\nSample of rural/urban status data:")
    print(rural_df.head())
    
    # Count rural vs urban hospitals
    if len(rural_df) > 0:
        rural_counts = rural_df['Rural vs. Urban'].value_counts()
        print(f"\nRural vs. Urban Hospital Counts:")
        for status, count in rural_counts.items():
            print(f"{status}: {count} hospitals")
