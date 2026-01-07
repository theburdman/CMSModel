from __future__ import annotations
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def extract_alpha_info(alpha_data: pd.DataFrame):
    """Extract hospital information datasets from ALPHA data.

    Returns the tuple: (date_df, names_df, zip_df, address_df, city_df, states_df, ccn_df, control_df, network_df)
    """
    # Get reporting start date
    date_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==100) & (alpha_data['Line Number']==2000)].copy()
    date_df = date_df.rename(columns={'Value':'Report Start Date'}).reset_index(drop=True)

    # Get hospital names
    names_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==100) & (alpha_data['Line Number']==300)].copy()
    names_df = names_df.rename(columns={'Value':'Hospital Name'}).reset_index(drop=True)

    # Get zip codes
    zip_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==300) & (alpha_data['Line Number']==200)].copy()
    zip_df = zip_df.rename(columns={'Value':'Zip Code'}).reset_index(drop=True)

    # Get hospital addresses
    address_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==100) & (alpha_data['Line Number']==100)].copy()
    address_df = address_df.rename(columns={'Value':'Address'}).reset_index(drop=True)

    # Get hospital cities
    city_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==100) & (alpha_data['Line Number']==200)].copy()
    city_df = city_df.rename(columns={'Value':'City'}).reset_index(drop=True)

    # Get state names
    states_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==200) & (alpha_data['Line Number']==200)].copy()
    states_df = states_df.rename(columns={'Value':'State'}).reset_index(drop=True)

    # Get CCN number and facility type
    ccn_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==200) & (alpha_data['Line Number']==300)].copy()
    ccn_df = ccn_df.rename(columns={'Value':'CCN'})
    ccn_df['Facility Code'] = ccn_df['CCN'].str[-4:-2]
    # Use pd.NA instead of np.nan to prevent DTypePromotionError (string vs float); ensure object dtype
    ccn_df['Facility Type'] = np.where(ccn_df['Facility Code'] == '13', 'CAH', pd.NA)
    ccn_df['Facility Type'] = ccn_df['Facility Type'].astype('object')
    ccn_df.reset_index(drop=True, inplace=True)

    # Get type of control
    control_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==100) & (alpha_data['Line Number']==2100)].copy()
    control_df = control_df.rename(columns={'Value':'Control Code'})
    control_map = {'1':'Non-Profit','2':'Non-Profit','3':'Private','4':'Private','5':'Private','6':'Private','7':'Government - Federal','8':'Government - City-County','9':'Government - County','10':'Government - State','11':'Government','12':'Government - City','13':'Government'}
    control_df['Type of Control'] = control_df['Control Code'].map(control_map); control_df.reset_index(drop=True,inplace=True)

    # Get hospital status (independent or network)
    network_df = alpha_data[(alpha_data['Worksheet Code']=='S200001') & (alpha_data['Column Number']==100) & (alpha_data['Line Number']==14100)].copy()
    network_df = network_df.rename(columns={'Value':'Network'}); network_df['Network']='Part of Network'; network_df.reset_index(drop=True,inplace=True)

    return date_df,names_df,zip_df,address_df,city_df,states_df,ccn_df,control_df,network_df

def extract_numeric_metrics(numeric_data: pd.DataFrame):
    """Extract numeric-derived hospital metrics and financials from NMRC data.

    Returns a tuple in this order:
    (
        beds_df, discharge_df, rural_df, charity_df, uncompensated_df,
        netIncome_df, interest_df, taxes_df, depreciation_df, rent_df, ebitda_df,
        liabilities_df, longLiabilities_df, operatingCosts_df, otherCosts_df,
        assets_df, reimburse_df, receivables_df, cash_df, provider_df
    )
    """
    # Beds
    beds_df = numeric_data[(numeric_data['Worksheet Code']=='S300001') & (numeric_data['Column Number'].astype(str)=='00200') & (numeric_data['Line Number']==100)].copy()
    beds_df = beds_df.rename(columns={'Value':'Number Beds'}).reset_index(drop=True)

    # Discharges
    discharge_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'E10A182') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 100)].copy()
    discharge_df = discharge_df.rename(columns={'Value': 'Discharges'}).reset_index(drop=True)

    # Rural vs Urban
    rural_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S200001') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 2600)].copy()
    rural_df = rural_df.rename(columns={'Value': 'Rural vs. Urban'}).reset_index(drop=True)
    rural_df['Rural vs. Urban'] = np.where(rural_df['Rural vs. Urban'] == 1, 'Urban', np.where(rural_df['Rural vs. Urban'] == 2, 'Rural', pd.NA))

    # Charity cost
    charity_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S100000') & (numeric_data['Column Number'].astype(str) == '00300') & (numeric_data['Line Number'] == 2300)].copy()
    charity_df = charity_df.rename(columns={'Value': 'Cost of Charity'}).reset_index(drop=True)

    # Uncompensated care cost
    uncompensated_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S100000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 3000)].copy()
    uncompensated_df = uncompensated_df.rename(columns={'Value': 'Cost of Uncompensated Care'}).reset_index(drop=True)

    # Net income
    netIncome_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G300000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 2900)].copy()
    netIncome_df = netIncome_df.rename(columns={'Value': 'Net Income'}).reset_index(drop=True)

    # Interest expense
    interest_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & (numeric_data['Column Number'].astype(str) == '01100') & (numeric_data['Line Number'] == 300)].copy()
    interest_df = interest_df.rename(columns={'Value': 'Interest'}).reset_index(drop=True)

    # Tax expense
    taxes_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & (numeric_data['Column Number'].astype(str) == '01300') & (numeric_data['Line Number'] == 300)].copy()
    taxes_df = taxes_df.rename(columns={'Value': 'Taxes'}).reset_index(drop=True)

    # Depreciation
    depreciation_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & (numeric_data['Column Number'].astype(str) == '00900') & (numeric_data['Line Number'] == 300)].copy()
    depreciation_df = depreciation_df.rename(columns={'Value': 'Depreciation'}).reset_index(drop=True)

    # Rent
    rent_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'A700003') & (numeric_data['Column Number'].astype(str) == '01000') & (numeric_data['Line Number'] == 300)].copy()
    rent_df = rent_df.rename(columns={'Value': 'Rent'}).reset_index(drop=True)

    # EBITDAR
    ebitda_df = interest_df[['Report Record Number', 'Interest', 'Year']].merge(taxes_df[['Report Record Number', 'Taxes', 'Year']], on = ['Report Record Number', 'Year'], how = 'outer')
    ebitda_df = ebitda_df.merge(depreciation_df[['Report Record Number', 'Depreciation', 'Year']], on = ['Report Record Number', 'Year'], how = 'outer')
    ebitda_df = ebitda_df.merge(rent_df[['Report Record Number', 'Rent', 'Year']], on = ['Report Record Number', 'Year'], how = 'outer')
    ebitda_df = ebitda_df.merge(netIncome_df[['Report Record Number', 'Net Income', 'Year']], on = ['Report Record Number', 'Year'], how = 'outer')
    ebitda_df = ebitda_df.fillna(0)
    ebitda_df['EBITDAR'] = ebitda_df['Net Income'] + ebitda_df['Interest'] + ebitda_df['Taxes'] + ebitda_df['Depreciation'] + ebitda_df['Rent']
    ebitda_df.reset_index(drop = True, inplace = True)

    # Liabilities (current)
    liabilities_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 4500)].copy()
    liabilities_df = liabilities_df.rename(columns={'Value': 'Total Current Liabilities'}).reset_index(drop=True)

    # Long-term liabilities
    longLiabilities_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 5000)].copy()
    longLiabilities_df = longLiabilities_df.rename(columns={'Value': 'Total Long-Term Liabilities'}).reset_index(drop=True)

    # Operating expense
    operatingCosts_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G300000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 400)].copy()
    operatingCosts_df = operatingCosts_df.rename(columns={'Value': 'Operating Expense'}).reset_index(drop=True)

    # Other expense
    otherCosts_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G300000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 2800)].copy()
    otherCosts_df = otherCosts_df.rename(columns={'Value': 'Other Expense'}).reset_index(drop=True)

    # Current assets
    assets_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 1100)].copy()
    assets_df = assets_df.rename(columns={'Value': 'Total Current Assets'}).reset_index(drop=True)

    # Government reimbursements
    reimburse_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S100000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 1800)].copy()
    reimburse_df = reimburse_df.rename(columns={'Value': 'Government Reimbursements'}).reset_index(drop=True)

    # Receivables components
    notes_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 300)].copy()
    accounts_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 400)].copy()
    other_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 500)].copy()
    allowances_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 600)].copy()

    notes_df = notes_df.rename(columns={'Value': 'Notes'})
    accounts_df = accounts_df.rename(columns={'Value': 'Accounts'})
    other_df = other_df.rename(columns={'Value': 'Other'})
    allowances_df = allowances_df.rename(columns={'Value': 'Allowances'})

    receivables_df = notes_df.merge(accounts_df[['Report Record Number', 'Year', 'Accounts']], on = ['Report Record Number', 'Year'], how = 'outer')
    receivables_df = receivables_df.merge(other_df[['Report Record Number', 'Year', 'Other']], on = ['Report Record Number', 'Year'], how = 'outer')
    receivables_df = receivables_df.merge(allowances_df[['Report Record Number', 'Year', 'Allowances']], on = ['Report Record Number', 'Year'], how = 'outer')

    receivables_df['Notes'] = receivables_df['Notes'].fillna(0)
    receivables_df['Accounts'] = receivables_df['Accounts'].fillna(0)
    receivables_df['Other'] = receivables_df['Other'].fillna(0)
    receivables_df['Allowances'] = receivables_df['Allowances'].fillna(0)

    receivables_df['Net Receivables'] = receivables_df['Notes'] + receivables_df['Accounts'] + receivables_df['Other'] - receivables_df['Allowances']

    # Cash on hand
    cash_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'G000000') & (numeric_data['Column Number'].astype(str) == '00100') & (numeric_data['Line Number'] == 100)].copy()
    cash_df = cash_df.rename(columns={'Value': 'Cash'}).reset_index(drop=True)

    # Provider types
    provider_df = numeric_data.loc[(numeric_data['Worksheet Code'] == 'S200001') & (numeric_data['Column Number'].astype(str) == '00400') & (numeric_data['Line Number'] == 300)].copy()
    provider_df = provider_df.rename(columns={'Value': 'Provider Type'}).reset_index(drop=True)

    logger.info(
        "Extracted numeric metrics: beds=%d, discharges=%d, rural=%d, charity=%d, uncomp=%d, ebitdar=%d",
        len(beds_df), len(discharge_df), len(rural_df), len(charity_df), len(uncompensated_df), len(ebitda_df)
    )

    return (
        beds_df, discharge_df, rural_df, charity_df, uncompensated_df,
        netIncome_df, interest_df, taxes_df, depreciation_df, rent_df, ebitda_df,
        liabilities_df, longLiabilities_df, operatingCosts_df, otherCosts_df,
        assets_df, reimburse_df, receivables_df, cash_df, provider_df
    )

def assemble_cms_df(alpha_info_tuple, numeric_metrics_tuple) -> pd.DataFrame:
    """Assemble the comprehensive CMS DataFrame using alpha/numeric component DataFrames.

    The merge order and keys follow the reference code provided.

    Args:
        alpha_info_tuple: Output of extract_alpha_info(alpha_data)
        numeric_metrics_tuple: Output of extract_numeric_metrics(numeric_data)

    Returns:
        cms_df (pd.DataFrame)
    """
    (
        date_df, names_df, zip_df, address_df, city_df, states_df, ccn_df, control_df, network_df
    ) = alpha_info_tuple

    (
        beds_df, discharge_df, rural_df, charity_df, uncompensated_df,
        netIncome_df, interest_df, taxes_df, depreciation_df, rent_df, ebitda_df,
        liabilities_df, longLiabilities_df, operatingCosts_df, otherCosts_df,
        assets_df, reimburse_df, receivables_df, cash_df, provider_df
    ) = numeric_metrics_tuple

    logger.info("Assembling CMS dataset from alpha and numeric components")

    cms_df = names_df[['Report Record Number', 'Hospital Name', 'Year']].merge(
        states_df[['Report Record Number', 'State']], on='Report Record Number', how='outer'
    )
    cms_df = cms_df.merge(city_df[['Report Record Number', 'City', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(address_df[['Report Record Number', 'Address', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(zip_df[['Report Record Number', 'Zip Code', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(date_df[['Report Record Number', 'Report Start Date', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(ccn_df[['Report Record Number', 'CCN', 'Facility Type', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(control_df[['Report Record Number', 'Type of Control', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(network_df[['Report Record Number', 'Network', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(beds_df[['Report Record Number', 'Number Beds', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(discharge_df[['Report Record Number', 'Discharges', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(rural_df[['Report Record Number', 'Rural vs. Urban', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(charity_df[['Report Record Number', 'Cost of Charity', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(uncompensated_df[['Report Record Number', 'Cost of Uncompensated Care', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(netIncome_df[['Report Record Number', 'Net Income', 'Year']], on=['Report Record Number', 'Year'], how='outer')

    # NEW: merge component columns before EBITDAR so they are preserved
    ebitdar_components_df = interest_df[['Report Record Number','Interest','Year']]\
        .merge(taxes_df[['Report Record Number','Taxes','Year']], on=['Report Record Number','Year'], how='outer')\
        .merge(depreciation_df[['Report Record Number','Depreciation','Year']], on=['Report Record Number','Year'], how='outer')\
        .merge(rent_df[['Report Record Number','Rent','Year']], on=['Report Record Number','Year'], how='outer')
    cms_df = cms_df.merge(ebitdar_components_df, on=['Report Record Number','Year'], how='outer')

    # Existing EBITDAR merge (retains prior calculation)
    cms_df = cms_df.merge(ebitda_df[['Report Record Number', 'EBITDAR', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(liabilities_df[['Report Record Number', 'Total Current Liabilities', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(longLiabilities_df[['Report Record Number', 'Total Long-Term Liabilities', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(cash_df[['Report Record Number', 'Cash', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(operatingCosts_df[['Report Record Number', 'Operating Expense', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(otherCosts_df[['Report Record Number', 'Other Expense', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(reimburse_df[['Report Record Number', 'Government Reimbursements', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(receivables_df[['Report Record Number', 'Net Receivables', 'Year']], on=['Report Record Number', 'Year'], how='outer')
    cms_df = cms_df.merge(assets_df[['Report Record Number', 'Total Current Assets', 'Year']], on=['Report Record Number', 'Year'], how='outer')

    logger.info("Assembled CMS dataset with shape %s", cms_df.shape)
    return cms_df

def assemble_master_incremental(master_df: pd.DataFrame, new_year_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Replace the specified year's rows in master CSV with newly assembled year data and write back to blob.

    - Reads current master from blob (if present)
    - Removes rows where Year == year
    - Assembles new year dataframe from ALPHA/NMRC blobs
    - Aligns columns to master schema
    - Writes updated master and a dated snapshot cms_data_MM_YY.csv
    Returns dict with counts and blob names.
    """
    # Drop existing rows for year and append new
    if 'Year' in master_df.columns:
        master_df = master_df[pd.to_numeric(master_df['Year'], errors='coerce') != year].copy()
    updated = pd.concat([master_df, new_year_df], ignore_index=True)
    return updated