from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import re, json, time, requests, zipfile
from io import BytesIO
from pathlib import Path
# Azure storage no longer needed for local version
import os, logging, requests, zipfile
from urllib.parse import urlencode
from io import BytesIO
from datetime import datetime  # ensure we have the class, not the module alias
from pathlib import Path
import csv
import uuid
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# LOCAL FILE STORAGE SETUP
# All files will be stored in the current working directory
print(f"Using current directory for data files: {os.getcwd()}")

# LOCAL FILE HELPER FUNCTIONS
def file_exists(filename):
    """Check if a file exists in the current directory."""
    return os.path.isfile(filename)

def read_csv_local(filename, **kwargs):
    """Read a CSV file from current directory into a pandas DataFrame."""
    try:
        return pd.read_csv(filename, **kwargs)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        raise

def write_csv_local(df, filename, **kwargs):
    """Write a pandas DataFrame to current directory as CSV."""
    try:
        df.to_csv(filename, **kwargs)
        print(f"Saved {filename} to local storage.")
    except Exception as e:
        print(f"Error writing {filename}: {e}")
        raise

def list_files_with_pattern(pattern):
    """List files in current directory matching a pattern (case-insensitive)."""
    matching_files = []
    for file in os.listdir('.'):
        if os.path.isfile(file) and pattern.lower() in file.lower():
            matching_files.append(file)
    return matching_files

def download_and_save_hcris_local(year):
    """Download HCRIS ZIP from CMS, extract, and save CSVs locally."""
    url = f'https://downloads.cms.gov/FILES/HCRIS/HOSP10FY{year}.ZIP'
    print(f'Downloading {url} ...')
    
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            print(f'Download failed status={resp.status_code}')
            return False
        
        print(f'Downloaded ZIP for year {year}')
        print('Extracting and saving to current directory...')
        
        # Extract ZIP in memory and save each CSV locally
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if name.lower().endswith('.csv'):
                    print(f' - Extracting {name}')
                    with open(name, 'wb') as f:
                        f.write(zf.read(name))
        
        print('HCRIS data saved locally.')
        return True
    except Exception as e:
        print(f'Error processing HCRIS data for year {year}: {e}')
        return False

# CONSTANTS AND FUNCTION DEFINITIONS

# Field extraction specifications
ALPHA_FIELD_SPECS = {
    'Report Start Date': ('S200001', 100, 2000),
    'Hospital Name': ('S200001', 100, 300),
    'Street Number': ('S200001', 100, 100),
    'City': ('S200001', 100, 200),
    'State': ('S200001', 200, 200),
    'Zip Code': ('S200001', 300, 200),
    'CCN': ('S200001', 200, 300),
    'Control Code': ('S200001', 100, 2100),
    'Network': ('S200001', 100, 14100)
}

NUMERIC_FIELD_SPECS = {
    'Number Beds': ('S300001', '00200', 100),
    'Discharges': ('E10A182', '00100', 100),
    'Rural vs. Urban': ('S200001', '00100', 2600),
    'Cost of Charity': ('S100000', '00300', 2300),
    'Cost of Uncompensated Care': ('S100000', '00100', 3000),
    'Net Income': ('G300000', '00100', 2900),
    'Interest': ('A700003', '01100', 300),
    'Taxes': ('A700003', '01300', 300),
    'Depreciation': ('A700003', '00900', 300),
    'Rent': ('A700003', '01000', 300),
    'Total Current Liabilities': ('G000000', '00100', 4500),
    'Total Long-Term Liabilities': ('G000000', '00100', 5000),
    'Operating Expense': ('G300000', '00100', 400),
    'Other Expense': ('G300000', '00100', 2800),
    'Total Current Assets': ('G000000', '00100', 1100),
    'Government Reimbursements': ('S100000', '00100', 1800),
    'Notes': ('G000000', '00100', 300),
    'Accounts': ('G000000', '00100', 400),
    'Other': ('G000000', '00100', 500),
    'Allowances': ('G000000', '00100', 600),
    'Cash': ('G000000', '00100', 100),
    'Provider Type': ('S200001', '00400', 300)
}

def extract_alpha_field(alpha_data, field_name, worksheet, column, line):
    """Extract a field from alpha data."""
    df = alpha_data.loc[
        (alpha_data['Worksheet Code'] == worksheet) & 
        (alpha_data['Column Number'] == column) & 
        (alpha_data['Line Number'] == line)
    ].copy()
    df = df.rename(columns={'Value': field_name})
    df.reset_index(drop=True, inplace=True)
    return df

def extract_numeric_field(numeric_data, field_name, worksheet, column, line):
    """Extract a field from numeric data."""
    df = numeric_data.loc[
        (numeric_data['Worksheet Code'] == worksheet) & 
        (numeric_data['Column Number'].astype(str) == column) & 
        (numeric_data['Line Number'] == line)
    ].copy()
    df = df.rename(columns={'Value': field_name})
    df.reset_index(drop=True, inplace=True)
    return df

def process_cms_data_extraction(alpha_data, numeric_data):
    """Extract and process all CMS data fields."""
    # Extract alpha fields
    extracted_fields = {}
    
    for field_name, (worksheet, column, line) in ALPHA_FIELD_SPECS.items():
        extracted_fields[field_name] = extract_alpha_field(alpha_data, field_name, worksheet, column, line)
    
    # Extract numeric fields
    for field_name, (worksheet, column, line) in NUMERIC_FIELD_SPECS.items():
        extracted_fields[field_name] = extract_numeric_field(numeric_data, field_name, worksheet, column, line)
    
    # CCN transformations
    ccn_df = extracted_fields['CCN']
    ccn_df['Facility Code'] = ccn_df['CCN'].str[-4:-2]
    ccn_df['Facility Type'] = np.where(ccn_df['Facility Code'] == '13', 'CAH', pd.NA)
    
    # Control code mapping
    control_df = extracted_fields['Control Code']
    control_dict = {
        '1': 'Non-Profit', '2': 'Non-Profit', '3': 'Private', '4': 'Private', '5': 'Private', '6': 'Private',
        '7': 'Government', '8': 'Government', '9': 'Government', '10': 'Government', '11': 'Government', '12': 'Government', '13': 'Government'
    }
    control_df['Type of Control'] = control_df['Control Code'].map(control_dict)
    
    # Network transformation
    network_df = extracted_fields['Network']
    network_df['Network'] = 'Part of Network'
    
    # Rural vs Urban transformation
    rural_df = extracted_fields['Rural vs. Urban']
    rural_df['Rural vs. Urban'] = np.where(rural_df['Rural vs. Urban'] == 1, 'Urban', 
                                          np.where(rural_df['Rural vs. Urban'] == 2, 'Rural', pd.NA))
    
    # Calculate EBITDAR
    ebitda_df = extracted_fields['Interest'][['Report Record Number', 'Interest', 'Year']].merge(
        extracted_fields['Taxes'][['Report Record Number', 'Taxes', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    ebitda_df = ebitda_df.merge(extracted_fields['Depreciation'][['Report Record Number', 'Depreciation', 'Year']], 
                               on=['Report Record Number', 'Year'], how='outer')
    ebitda_df = ebitda_df.merge(extracted_fields['Rent'][['Report Record Number', 'Rent', 'Year']], 
                               on=['Report Record Number', 'Year'], how='outer')
    ebitda_df = ebitda_df.merge(extracted_fields['Net Income'][['Report Record Number', 'Net Income', 'Year']], 
                               on=['Report Record Number', 'Year'], how='outer')
    ebitda_df = ebitda_df.fillna(0)
    ebitda_df['EBITDAR'] = ebitda_df['Net Income'] + ebitda_df['Interest'] + ebitda_df['Taxes'] + ebitda_df['Depreciation'] + ebitda_df['Rent']
    ebitda_df.reset_index(drop=True, inplace=True)
    
    # Calculate Net Receivables
    receivables_df = extracted_fields['Notes'].merge(
        extracted_fields['Accounts'][['Report Record Number', 'Year', 'Accounts']], on=['Report Record Number', 'Year'], how='outer')
    receivables_df = receivables_df.merge(
        extracted_fields['Other'][['Report Record Number', 'Year', 'Other']], on=['Report Record Number', 'Year'], how='outer')
    receivables_df = receivables_df.merge(
        extracted_fields['Allowances'][['Report Record Number', 'Year', 'Allowances']], on=['Report Record Number', 'Year'], how='outer')
    
    receivables_df['Notes'] = receivables_df['Notes'].fillna(0)
    receivables_df['Accounts'] = receivables_df['Accounts'].fillna(0)
    receivables_df['Other'] = receivables_df['Other'].fillna(0)
    receivables_df['Allowances'] = receivables_df['Allowances'].fillna(0)
    receivables_df['Net Receivables'] = (receivables_df['Notes'] + receivables_df['Accounts'] + 
                                        receivables_df['Other'] - receivables_df['Allowances'])
    
    # Assemble final dataframe
    cms_df = extracted_fields['Hospital Name'][['Report Record Number', 'Hospital Name', 'Year']].merge(
        extracted_fields['State'][['Report Record Number', 'State']], on='Report Record Number', how='outer')
    
    merge_fields = [
        ('City', ['City', 'Year']), ('Street Number', ['Street Number', 'Year']), ('Zip Code', ['Zip Code', 'Year']),
        ('Report Start Date', ['Report Start Date', 'Year']), ('Number Beds', ['Number Beds', 'Year']),
        ('Discharges', ['Discharges', 'Year']), ('Cost of Charity', ['Cost of Charity', 'Year']),
        ('Cost of Uncompensated Care', ['Cost of Uncompensated Care', 'Year']), ('Net Income', ['Net Income', 'Year']),
        ('Total Current Liabilities', ['Total Current Liabilities', 'Year']), 
        ('Total Long-Term Liabilities', ['Total Long-Term Liabilities', 'Year']),
        ('Cash', ['Cash', 'Year']), ('Operating Expense', ['Operating Expense', 'Year']),
        ('Other Expense', ['Other Expense', 'Year']), ('Government Reimbursements', ['Government Reimbursements', 'Year']),
        ('Total Current Assets', ['Total Current Assets', 'Year'])
    ]
    
    for field_name, cols in merge_fields:
        cms_df = cms_df.merge(extracted_fields[field_name][['Report Record Number'] + cols], 
                             on=['Report Record Number', 'Year'], how='outer')
    
    # Merge special processed fields
    cms_df = cms_df.merge(ccn_df[['Report Record Number', 'CCN', 'Facility Type', 'Year']], 
                         on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(control_df[['Report Record Number', 'Type of Control', 'Year']], 
                         on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(network_df[['Report Record Number', 'Network', 'Year']], 
                         on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(rural_df[['Report Record Number', 'Rural vs. Urban', 'Year']], 
                         on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(ebitda_df[['Report Record Number', 'EBITDAR', 'Year']], 
                         on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(receivables_df[['Report Record Number', 'Net Receivables', 'Year']], 
                         on=['Report Record Number', 'Year'], how='outer')
    
    # Tag EBITDAR > $400k
    cms_df['>$400k EBITDAR'] = np.where(cms_df['EBITDAR'] > 400000, 1, 0)

    # Tag debt < $5M or < 5x net income
    avgNI_df = cms_df.copy()
    avgNI_df = avgNI_df.groupby('Hospital Name')['Net Income'].mean()
    avgNI_df = pd.DataFrame(avgNI_df)
    avgNI_df.reset_index(inplace = True)
    avgNI_df.rename(columns = {'Net Income': 'Net Income Average'}, inplace = True)

    cms_df = cms_df.merge(avgNI_df, on = 'Hospital Name', how = 'left')
    cms_df['Total Debt'] = cms_df['Total Current Liabilities'] + cms_df['Total Long-Term Liabilities']

    cms_df['Debt <$5M or 5x NI'] = np.where((cms_df['Total Debt'] < 5000000) | (cms_df['Total Debt'] < 5 * cms_df['Net Income Average']) , 1, 0)

    # Tag positive cash position hospitals
    cms_df['Positive Cash Position'] = np.where(cms_df['Cash'] > 0, 1, 0)

    # Tag hospitals with financial reserves greater than 65 days of operating expenses
    cms_df['Cash vs. 65 Days'] = np.where(cms_df['Cash'] > cms_df['Operating Expense'] / 365 * 65, 1, 0)

    # Tag hospitals with net income greater than 3% total expenses
    cms_df['Total Expenses'] = cms_df['Operating Expense'] + cms_df['Other Expense']
    cms_df['Income vs. 3% Total Expenses'] = np.where(cms_df['Net Income'] > .03 * cms_df['Total Expenses'], 1, 0)

    # Tag hospitals with total current assets > 100 days of operating expenses
    cms_df['Assets vs. 100 Days'] = np.where(cms_df['Total Current Assets'] > cms_df['Operating Expense'] / 365 * 100, 1, 0)

    # Tag hospitals receiving at least $100k in government reimbursements per year
    cms_df['$100k Reimbursement'] = np.where(cms_df['Government Reimbursements'] > 100000, 1, 0)

    # Tag hospitals with net receivables less than 88 days of operating expenses
    cms_df['Net Receivables vs. 88 Days'] = np.where(cms_df['Net Receivables'] > cms_df['Operating Expense'] / 365 * 88, 1, 0)

    return cms_df

def run_cms_pipeline():
	"""
	Main function that executes the entire CMS data pipeline.
	This function is called by the Azure Function timer trigger.
	"""
	import logging
	
	logging.info("Starting CMS pipeline execution...")
	
	# STEP 1 - IF MASTER CMS FILE IS NOT FOUND, ASSEMBLE MASTER DATASET; OTHERWISE SKIP
	MASTER_PATH = "cms_master.csv"
	if file_exists(MASTER_PATH):
		print(f"{MASTER_PATH} found locally; skipping master assembly.")
		cms_df = read_csv_local(MASTER_PATH, dtype={'CCN': 'string'})

	else:
		print(f"{MASTER_PATH} not found; assembling master...")
	
		# Determine end_year: latest available HCRIS year or fallback one year earlier
		latest_year = datetime.utcnow().year - 1
		latest_url = f"https://downloads.cms.gov/FILES/HCRIS/HOSP10FY{latest_year}.ZIP"
		try:
			available = requests.head(latest_url, timeout=15).status_code == 200
		except Exception:
			available = False
		end_year = latest_year if available else latest_year - 1
		print(f"Resolved end_year={end_year} (latest_year={latest_year}, available={available})")

		# Download and extract selected year to blob storage
		download_and_save_hcris_local(end_year)

		# Assemble all data available for cms dataframe
		start_year = 2015

		# Loop through available data and create alpha dataframe
		alpha_data = pd.DataFrame()
		for i in range(start_year, end_year + 1):
			pattern = f'hosp10_{i}_alpha.csv'
			# Find matching file (case-insensitive)
			matching_files = list_files_with_pattern(pattern)
			if not matching_files:
				raise FileNotFoundError(f"No file found matching {pattern} (case-insensitive)")
			
			file_name = matching_files[0]
			alpha = read_csv_local(file_name, header=None, dtype={'CCN': 'string'}, names=[
				'Report Record Number', 'Worksheet Code', 'Line Number', 'Column Number', 'Value'
			])
			alpha['Form ID'] = (
				alpha['Worksheet Code'] + ',' +
				alpha['Line Number'].astype(str) + ',' +
				alpha['Column Number'].astype(str)
			)
			alpha['Year'] = i
			alpha_data = pd.concat([alpha_data, alpha], ignore_index=True)

		# Loop through available data and create numeric dataframe
		numeric_data = pd.DataFrame()

		for i in range(start_year, end_year + 1):
			pattern = f'hosp10_{i}_nmrc.csv'
			# Find matching file (case-insensitive)
			matching_files = list_files_with_pattern(pattern)
			if not matching_files:
				raise FileNotFoundError(f"No file found matching {pattern} (case-insensitive)")
			
			file_name = matching_files[0]
			numeric = read_csv_local(file_name, header=None, dtype={'CCN': 'string'}, names=[
				'Report Record Number', 'Worksheet Code', 'Line Number', 'Column Number', 'Value'
			])
			numeric['Form ID'] = (
				numeric['Worksheet Code'] + ',' +
				numeric['Line Number'].astype(str) + ',' +
				numeric['Column Number'].astype(str)
			)
			numeric['Year'] = i
			numeric_data = pd.concat([numeric_data, numeric], ignore_index=True)

		cms_df = process_cms_data_extraction(alpha_data, numeric_data)
		print("Initial CMS data compilation complete.")
		# ---- END REFACTORED SECTION ----

		# Write baseline cms CSV locally
		write_csv_local(cms_df, 'cms_master.csv', index=False)
		print("Created master CMS file locally.")

	# STEP 2 - GET LATEST DATA FROM CMS.GOV AND PROCESS

	# Determine end_year: latest available HCRIS year or fallback one year earlier
	latest_year = datetime.utcnow().year - 1
	latest_url = f"https://downloads.cms.gov/FILES/HCRIS/HOSP10FY{latest_year}.ZIP"
	try:
		available = requests.head(latest_url, timeout=15).status_code == 200
	except Exception:
		available = False
	end_year = latest_year if available else latest_year - 1
	print(f"Resolved end_year={end_year} (latest_year={latest_year}, available={available})")

	# Download and extract selected year to blob storage
	download_and_save_hcris_local(end_year)

	# Process most recent file locally
	alpha_pattern = f"hosp10_{latest_year}_alpha.csv"
	alpha_files = list_files_with_pattern(alpha_pattern)
	if not alpha_files:
		raise FileNotFoundError(f"No file found matching {alpha_pattern}")
	alpha = read_csv_local(alpha_files[0], header=None, dtype={'CCN': 'string'}, names=['Report Record Number', 'Worksheet Code', 'Line Number', 'Column Number', 'Value'])
	alpha['Form ID'] = (alpha['Worksheet Code'] + ',' + alpha['Line Number'].astype(str) + ',' + alpha['Column Number'].astype(str))
	alpha['Year'] = latest_year

	numeric_pattern = f"hosp10_{latest_year}_nmrc.csv"
	numeric_files = list_files_with_pattern(numeric_pattern)
	if not numeric_files:
		raise FileNotFoundError(f"No file found matching {numeric_pattern}")
	numeric = read_csv_local(numeric_files[0], header=None, dtype={'CCN': 'string'}, names=['Report Record Number', 'Worksheet Code', 'Line Number', 'Column Number', 'Value'])
	numeric['Form ID'] = (numeric['Worksheet Code'] + ',' + numeric['Line Number'].astype(str) + ',' + numeric['Column Number'].astype(str))
	numeric['Year'] = latest_year

	cms_recent_df = process_cms_data_extraction(alpha, numeric)
	print("Latest CMS data compilation complete.")

	# STEP 3 - COMBINE MOST RECENT YEAR AND PRIOR YEAR FOR FURTHER ANALYSIS, KEEP MASTER UP-TO-DATE, AND CREATE DEDUPE FILE
	if file_exists(MASTER_PATH):
		print(f'{MASTER_PATH} found locally; loading existing master.')
		cms_df = read_csv_local(MASTER_PATH, dtype={'CCN': 'string'})
	else:
		pass

	cms_df = cms_df[~cms_df['Year'].isin(pd.to_numeric(cms_recent_df['Year'], errors='coerce').dropna().astype(int))] # Drop data that is from same year as newly downloaded data to ensure newest data

	recent_year = cms_recent_df['Year'].iloc[0] # Extract year data from files to match most recent two
	master_max_year = cms_df['Year'].max()

	if master_max_year == recent_year - 1:  # Check if they are consecutive (master year should be recent_year - 1)
		master_last_year_df = cms_df[cms_df['Year'] == master_max_year]  # Filter master to include ONLY the last year
		cms_dedupe_df = pd.concat([master_last_year_df, cms_recent_df], ignore_index=True) # Combine data
		cms_dedupe_df = cms_dedupe_df.drop_duplicates(subset='CCN', keep='last')  # Remove duplicates and keep newest record
	else:
		cms_dedupe_df = cms_recent_df.copy()  # Do not combine if not consecutive years
		print("Year mismatch: not combining because the datasets are not consecutive.")

	cms_df = pd.concat([cms_df, cms_recent_df], ignore_index=True) # Update master dataframe
	write_csv_local(cms_df, 'cms_master.csv', index=False) # Write copy of all updated data locally
	timestamp = datetime.now().strftime("%m_%Y") # Create timestamp for snapshot version of cms master
	write_csv_local(cms_df, f'cms_master_{timestamp}.csv', index=False) # Write timestamped copy of master snapshot locally
	cms_dedupe_df = cms_dedupe_df.drop_duplicates(subset='CCN') # Dedupe on the CCN records
	write_csv_local(cms_dedupe_df, 'cms_dedupe.csv', index=False) # Write copy of deduped, recent two years of data locally
	print('Master CMS data updated locally; dedupe file created.')

	# STEP 4 - GET GEOCODE DATA FOR DEDUPE SET AND CHECK CAH STATUS
	print('Getting geocode data.')
	# Load geocode data locally (create empty if doesn't exist)
	if file_exists('geocode_data.csv'):
		geocodes_df = read_csv_local('geocode_data.csv', dtype={'CCN': 'string'})
	else:
		geocodes_df = pd.DataFrame(columns=['CCN', 'Latitude', 'Longitude'])
		write_csv_local(geocodes_df, 'geocode_data.csv', index=False)

	cms_dedupe_df = cms_dedupe_df.merge(geocodes_df[['CCN', 'Latitude', 'Longitude']],on="CCN",how="left") # Attach geocode data to dedupe list
	need_geocodes_df = cms_dedupe_df[cms_dedupe_df['Latitude'].isna() | cms_dedupe_df['Longitude'].isna()] # Create list of hospitals that need geocodes
	need_geocodes_df = need_geocodes_df[['CCN', 'Street Number', 'City', 'State', 'Zip Code']]
	need_geocodes_df['Full Address'] = need_geocodes_df['Street Number'].astype(str) + ', ' + need_geocodes_df['City'].astype(str) + ', ' + need_geocodes_df['State'].astype(str) + ' ' + need_geocodes_df['Zip Code'].astype(str)
	write_csv_local(need_geocodes_df, 'need_geocodes.csv', index=False) # Write list of hospitals that need geocodes locally

	if need_geocodes_df.empty: # Check if we need to get geocodes
		print('No CCNs to geocode. Skipping API calls.')
	else:
		print(f'{len(need_geocodes_df)} CCNs need geocoding. Proceeding with API calls.')
	
		GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
		if not GOOGLE_API_KEY:
			raise ValueError('GOOGLE_API_KEY environment variable is required for geocoding')
		GEOCODE_CALLS = 'raw_geocode_calls.csv'

		print('Calling Google Geocoding API')
		# Initialize raw_geocode_calls.csv locally with headers
		raw_calls_df = pd.DataFrame(columns=['CCN', 'Latitude', 'Longitude'])
		write_csv_local(raw_calls_df, GEOCODE_CALLS, index=False)

		def geocode_address(address):
			'''Call Google Geocoding API for a single address.'''
			base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
			params = {'address': address, 'key': GOOGLE_API_KEY}
			url = f'{base_url}?{urlencode(params)}'

			response = requests.get(url)
			data = response.json()

			if data['status'] != 'OK':
				return None, None  # Failed lookup

			location = data['results'][0]['geometry']['location']
			return location['lat'], location['lng']

		results = []

		total_calls = len(need_geocodes_df)
		calls_made = 0

		for idx, row in need_geocodes_df.iterrows():
			address = row['Full Address']

			lat, lng = geocode_address(address)
			calls_made += 1
			remaining = total_calls - calls_made

			print(f'{calls_made} calls made, {remaining} remaining')

			# Collect result to write to blob after loop
			results.append({'CCN': row['CCN'], 'Latitude': lat, 'Longitude': lng})

			time.sleep(0.05)  # 5 calls/sec – safe for small batches

		# Write all geocode results locally
		raw_calls_df = pd.DataFrame(results)
		write_csv_local(raw_calls_df, GEOCODE_CALLS, index=False)
	
		# Read files locally
		raw_calls_df = read_csv_local('raw_geocode_calls.csv', dtype={'CCN': 'string'})
		need_geocodes_df = read_csv_local('need_geocodes.csv', dtype={'CCN': 'string'})
		geocodes_df = read_csv_local('geocode_data.csv', dtype={'CCN': 'string'})

		geocodes_df = pd.concat([geocodes_df, raw_calls_df], ignore_index=True) # Attach new geocode data to the existing file
		geocodes_df = geocodes_df.drop_duplicates(subset='CCN', keep='last') # Remove duplicates as a safety step
		geocodes_df = geocodes_df.sort_values('CCN')
		write_csv_local(geocodes_df, 'geocode_data.csv', index=False) # Overwrite old geocode data locally
		processed_ccns = set(geocodes_df['CCN'])
		need_geocodes_df = need_geocodes_df[~need_geocodes_df['CCN'].isin(processed_ccns)]
		write_csv_local(need_geocodes_df, 'need_geocodes.csv', index=False)
		raw_calls_df = raw_calls_df[~raw_calls_df['CCN'].isin(processed_ccns)]
		write_csv_local(raw_calls_df, 'raw_geocode_calls.csv', index=False)

	cms_deupe_df = cms_dedupe_df.merge(geocodes_df[['CCN', 'Latitude', 'Longitude']],on='CCN',how='left') # Attach geocode data to dedupe list

	# STEP 5 - DETERMINE CAH ELIGIBILITY
	print('Determining CAH eligibility for dedupe set.')


	def haversine_np(lat1, lon1, lat2, lon2): # Vectorized Haversine  
		'''Vectorized Haversine distance in miles. All args must be numpy arrays, same shape.'''
		lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
		dlat = lat2 - lat1
		dlon = lon2 - lon1

		a = (np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2)

		c = 2 * np.arcsin(np.sqrt(a))
		miles = 3958.8 * c
		return miles

	# Load existing distance calculations if they exist locally
	if file_exists('inter_hospital_distances.csv'):
		dist_df = read_csv_local(
			'inter_hospital_distances.csv',
			dtype={'CCN1': 'string', 'CCN2': 'string'}
		)
	else:
		dist_df = pd.DataFrame(columns=['CCN1', 'CCN2', 'Distance'])

	# Prepare data for processing
	coords_df = cms_dedupe_df[['CCN', 'Latitude', 'Longitude']].copy()
	coords_df['CCN'] = coords_df['CCN'].astype(str).str.strip().str.zfill(6)  # Clean and pad CCNs
	coords_df['Latitude'] = pd.to_numeric(coords_df['Latitude'], errors='coerce')
	coords_df['Longitude'] = pd.to_numeric(coords_df['Longitude'], errors='coerce')
	coords_df = coords_df.dropna(subset=['Latitude', 'Longitude'])  # Drop missing coords
	coords_df = coords_df.drop_duplicates(subset='CCN')  # Remove duplicates if any

	# Generate all pairwise combinations of CCNs
	all_pairs = pd.DataFrame(
	    list(combinations(coords_df['CCN'], 2)),
	    columns=['CCN1', 'CCN2']
	)
	# Sort CCN1, CCN2 so (A,B) matches (B,A) in existing distances
	all_pairs[['CCN1','CCN2']] = np.sort(all_pairs[['CCN1','CCN2']], axis=1)

	# Also sort dist_df for comparison
	dist_df[['CCN1','CCN2']] = np.sort(dist_df[['CCN1','CCN2']], axis=1)

	# Identify missing pairs
	merged = all_pairs.merge(dist_df[['CCN1', 'CCN2']],on=['CCN1', 'CCN2'],how='left',indicator=True)
	missing_pairs = merged[merged['_merge'] == 'left_only'][['CCN1','CCN2']].reset_index(drop=True)
	print(f'Missing distance pairs needing calculation: {len(missing_pairs)}')

	# Calculate
	if len(missing_pairs) > 0:

		mp = (missing_pairs
			.merge(coords_df, left_on='CCN1', right_on='CCN', how='left')
			.merge(coords_df, left_on='CCN2', right_on='CCN', how='left',
				suffixes=('_1','_2'))
		)

		lat1 = mp['Latitude_1'].to_numpy()
		lon1 = mp['Longitude_1'].to_numpy()
		lat2 = mp['Latitude_2'].to_numpy()
		lon2 = mp['Longitude_2'].to_numpy()

		distances = np.round(haversine_np(lat1, lon1, lat2, lon2), 1)  # Vectorized calculation

		new_rows = pd.DataFrame({
			'CCN1': mp['CCN1'],
			'CCN2': mp['CCN2'],
			'Distance': distances
		})

		dist_df = pd.concat([dist_df, new_rows], ignore_index=True)
		write_csv_local(dist_df, 'inter_hospital_distances.csv', index=False)
		print(f'Appended {len(new_rows)} new distance records locally.')
	else:
		print('Distances computed.')

	# Compute minimum neighbor distances
	dist = read_csv_local('inter_hospital_distances.csv', dtype={'CCN1':'string','CCN2':'string'})
	dist['Distance'] = pd.to_numeric(dist['Distance'], errors='coerce')
	valid_dist = dist[dist['Distance'] > 1]  # Exclude same-campus facilities

	# Stack both directions for min distance calculation
	a = valid_dist[['CCN1', 'CCN2', 'Distance']].rename(columns={'CCN1':'CCN'})
	b = valid_dist[['CCN2', 'CCN1', 'Distance']].rename(columns={'CCN2':'CCN'})
	all_pairs_stacked = pd.concat([a, b], ignore_index=True)

	min_dist = all_pairs_stacked.groupby('CCN')['Distance'].min().reset_index()
	min_dist.rename(columns={'Distance':'MinNeighborDistance'}, inplace=True)

	# Normalize formatting
	cms_dedupe_df['CCN'] = cms_dedupe_df['CCN'].astype(str).str.strip().str.zfill(6)
	min_dist['CCN'] = min_dist['CCN'].astype(str).str.strip().str.zfill(6)
	cms_dedupe_df = cms_dedupe_df.merge(min_dist, how='left', on='CCN')

	# Flag any missing neighbors
	cms_dedupe_df['HasNeighbors'] = cms_dedupe_df['MinNeighborDistance'].notna().astype(int)
	missing_neighbor_count = (cms_dedupe_df['HasNeighbors'] == 0).sum()
	if missing_neighbor_count > 0:
		print(f"{missing_neighbor_count} CCNs found with NO neighbors in inter-hospital distance calculation.")
	else:
		print("All CCNs have at least one calculated neighbor distance.")

	cms_dedupe_df['MinNeighborDistance'] = cms_dedupe_df['MinNeighborDistance'].fillna(9999)

	# Determine CAH eligibility
	cms_dedupe_df['EligibleForCAH'] = (
		(cms_dedupe_df['MinNeighborDistance'] > 35) &
		(cms_dedupe_df['Facility Type'] != 'CAH') &
		(cms_dedupe_df['Rural vs. Urban'].str.lower() == 'rural')
	).astype(int)

	write_csv_local(cms_dedupe_df, 'cms_dedupe.csv', index=False)

	# STEP 6A - BUILD ML MODEL TO TEST VALIDITY = AUROC SCORE = 0.8602

	# df = pd.read_csv('cms_master.csv') # Load data
	# df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
	# df = df.sort_values(['ccn', 'year']) # Sort to make time series
	# df['ltl_next_year'] = df.groupby('ccn')['total_long-term_liabilities'].shift(-1) # Create target variable
	# df['built_project'] = (df['ltl_next_year'] - df['total_long-term_liabilities'] >= 10_000_000).astype(int)
	# df = df.dropna(subset=['ltl_next_year']) # Drop recent years since no later data exists yet
	# categorical_cols = ['facility_type','type_of_control','network','rural_vs._urban']

	# numeric_cols = [col for col in df.columns
	#     if col not in categorical_cols
	#     and col not in ['built_project','ltl_next_year','total_long-term_liabilities',
	#         'report_start_date','hospital_name','state','city','street_number','zip_code']
	#     and df[col].dtype != 'object']

	# gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
	# train_idx, test_idx = next(gss.split(df, groups=df['ccn']))
	# train_df = df.iloc[train_idx]
	# test_df = df.iloc[test_idx]

	# preprocess = ColumnTransformer(
	#     transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),('num', 'passthrough', numeric_cols)])

	# model = Pipeline(steps=[
	#     ('preprocess', preprocess),
	#     ('clf', RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_split=4,random_state=42))])

	# X_train = train_df[categorical_cols + numeric_cols]
	# y_train = train_df['built_project']

	# X_test = test_df[categorical_cols + numeric_cols]
	# y_test = test_df['built_project']

	# model.fit(X_train, y_train)

	# test_pred_prob = model.predict_proba(X_test)[:, 1]
	# auc = roc_auc_score(y_test, test_pred_prob)

	# print(f'Test AUC: {auc:.4f}')

	# from sklearn.metrics import roc_curve, auc
	# import matplotlib.pyplot as plt

	# fpr, tpr, thresholds = roc_curve(y_test, test_pred_prob)

	# roc_auc = auc(fpr, tpr)
	# print(f'AUROC Score: {roc_auc:.4f}')

	# plt.figure(figsize=(8, 6))
	# plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
	# plt.plot([0, 1], [0, 1], linestyle='--', label='Chance Level')
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver Operating Characteristic (ROC)')
	# plt.legend(loc='lower right')
	# plt.grid(True)
	# plt.show()

	# test_df['predicted_probability'] = test_pred_prob

	# STEP 6B - BUILD PRODUCTION MODEL AND PREDICT ON DEDUPE SET
	cms_master_df = read_csv_local('cms_master.csv')
	cms_dedupe_df = read_csv_local('cms_dedupe.csv')

	# Standardize column names
	cms_master_df.columns = cms_master_df.columns.str.strip().str.replace(' ', '_').str.lower()
	cms_dedupe_df.columns = cms_dedupe_df.columns.str.strip().str.replace(' ', '_').str.lower()

	model_df = cms_master_df.copy()
	model_df = model_df.sort_values(['ccn', 'year'])

	model_df['ltl_next_year'] = model_df.groupby('ccn')['total_long-term_liabilities'].shift(-1)
	model_df['built_project'] = (model_df['ltl_next_year'] - model_df['total_long-term_liabilities'] >= 10_000_000).astype(int)

	model_df = model_df.dropna(subset=['ltl_next_year'])
	categorical_cols = ['facility_type','type_of_control','network','rural_vs._urban']
	numeric_cols = [col for col in model_df.columns if col not in categorical_cols and col not in ['built_project','ltl_next_year','total_long-term_liabilities',
	        'report_start_date','hospital_name','state','city','street_number','zip_code'] and model_df[col].dtype != 'object']

	dedupe_years = cms_dedupe_df['year'].unique() # Remove prediction years from training data
	train_df = model_df[~model_df['year'].isin(dedupe_years)]
	preprocess = ColumnTransformer(transformers=[('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
	            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols),('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),]), numeric_cols)])
	# Build model
	model = Pipeline(steps=[('preprocess', preprocess),('clf', RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_split=4,random_state=42))])
	# Fit model
	X_train = train_df[categorical_cols + numeric_cols]
	y_train = train_df['built_project']
	model.fit(X_train, y_train)

	X_score = cms_dedupe_df[categorical_cols + numeric_cols]
	score_pred_prob = model.predict_proba(X_score)[:, 1]

	cms_dedupe_df['predicted_probability'] = score_pred_prob # Attach prediction values 
	write_csv_local(cms_dedupe_df, 'cms_visuals.csv', index=False) # Write output locally
	print('CMS production model run complete. All outputs saved locally.')

if __name__ == "__main__":
	print("=" * 60)
	print("CMS HCRIS Data Pipeline - Local Version")
	print("=" * 60)
	run_cms_pipeline()
	print("=" * 60)
	print("Pipeline complete!")
	print("=" * 60)