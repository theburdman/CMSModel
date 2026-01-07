# CMS HCRIS Data Processing Azure Function

This Azure Function processes CMS Hospital Cost Report Information System (HCRIS) data monthly, performing data extraction, geocoding, CAH eligibility determination, and predictive modeling.

## Function Details

- **Function Name**: `korteCMSFunction`
- **Resource Group**: `RGKorteNext`
- **Schedule**: Monthly on the 1st at 6:00 PM UTC
- **Runtime**: Python 3.10+

## Features

1. **Data Acquisition**: Downloads latest HCRIS data from CMS.gov
2. **Data Processing**: Extracts and transforms hospital financial and operational data
3. **Geocoding**: Uses Google Maps API to geocode hospital addresses
4. **CAH Eligibility**: Determines Critical Access Hospital eligibility based on location
5. **Predictive Modeling**: ML model to predict hospital capital projects
6. **Blob Storage**: All data stored in Azure Blob Storage (`cmsdata` account)

## Prerequisites

- Azure Subscription
- Azure Functions Core Tools (for local development)
- Python 3.10 or higher
- Azure Storage Account (`cmsdata`)
- Google Maps Geocoding API Key

## Pre-Deployment Testing

### Quick Syntax Check (No Azure Connection)

Before deploying, verify your code has no syntax or indentation errors:

```bash
python3 check_syntax.py
```

This will check:
- ✓ Python syntax
- ✓ Indentation consistency  
- ✓ Import statements

### Local Pipeline Test (Requires Azure Connection)

To test the actual pipeline locally before deploying:

```bash
python3 test_local.py
```

**Requirements:**
- Azure Storage connection string set in environment variable
- Google API key set in environment variable
- Or hardcoded values in `master.py` (for testing only)

This will:
1. Connect to your Azure Blob Storage
2. Run the complete CMS pipeline
3. Save outputs to blob storage
4. Generate a log file: `cms_pipeline_test.log`

## Local Development Setup

1. **Install Azure Functions Core Tools**:
   ```bash
   brew install azure-functions-core-tools@4  # macOS
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure local settings**:
   - Copy `local.settings.json.example` to `local.settings.json`
   - Add your Azure Storage connection string
   - Add your Google API key

5. **Run locally**:
   ```bash
   func start
   ```

## Deployment to Azure

### Option 1: Using Azure CLI

```bash
# Login to Azure
az login

# Create Function App (if not exists)
az functionapp create \
  --resource-group RGKorteNext \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.10 \
  --functions-version 4 \
  --name korteCMSFunction \
  --storage-account cmsdata \
  --os-type Linux

# Deploy the function
func azure functionapp publish korteCMSFunction
```

### Option 2: Using VS Code

1. Install Azure Functions extension
2. Right-click on project folder
3. Select "Deploy to Function App"
4. Choose `korteCMSFunction`

## Configuring Secrets in Azure Portal

After deployment, configure application settings:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Resource Groups** → **RGKorteNext** → **korteCMSFunction**
3. Click **Configuration** in the left menu
4. Under **Application settings**, click **+ New application setting**
5. Add these settings:

   | Name | Value |
   |------|-------|
   | `AZURE_STORAGE_CONNECTION_STRING` | Your Azure Storage connection string |
   | `GOOGLE_API_KEY` | Your Google Maps API key |

6. Click **Save** at the top

## Testing the Function

### Test with HTTP Trigger (Manual)

The function includes an HTTP endpoint for manual testing:

```bash
# Get the function URL from Azure Portal
curl -X POST "https://kortecmsfunction.azurewebsites.net/api/cms_manual_trigger?code=YOUR_FUNCTION_KEY"
```

### Test Schedule (Frequent for Testing)

To test more frequently during development, update the schedule in `function_app.py`:

```python
@app.schedule(
    schedule="0 */5 * * * *",  # Every 5 minutes for testing
    # schedule="0 0 18 1 * *",  # Monthly (production)
    ...
)
```

## Cron Schedule Examples

- **Every 5 minutes** (testing): `0 */5 * * * *`
- **Every hour**: `0 0 * * * *`
- **Daily at 6 PM**: `0 0 18 * * *`
- **Monthly on 1st at 6 PM**: `0 0 18 1 * *` (production)

Format: `{second} {minute} {hour} {day} {month} {day-of-week}`

## Monitoring

View logs and execution history:

1. Azure Portal → korteCMSFunction
2. Click **Monitor** or **Log stream** in the left menu
3. View Application Insights for detailed metrics

## Data Outputs (Azure Blob Storage)

All files are stored in the `cmsfiles` container:

- `cms_master.csv` - Complete historical dataset
- `cms_master_MM_YYYY.csv` - Monthly snapshots
- `cms_dedupe.csv` - Recent deduplicated data
- `cms_visuals.csv` - Final output with predictions
- `geocode_data.csv` - Geocoded addresses
- `inter_hospital_distances.csv` - Hospital distance matrix
- `HOSP10_{year}_*.csv` - HCRIS raw data files

## Troubleshooting

### Function Timeout
The function has a 60-minute timeout. If processing takes longer, consider:
- Splitting into multiple functions
- Increasing timeout in `host.json` (Premium plan required)

### API Rate Limits
- Google Geocoding API: 50 requests/second (default)
- Adjust `time.sleep()` in geocoding section if needed

### Memory Issues
- Processing large datasets requires adequate memory
- Consider Premium or Dedicated App Service Plan for larger workloads

## Project Structure

```
CMSTimerIncremental/
├── function_app.py          # Azure Function definitions
├── master.py                # Main pipeline logic
├── requirements.txt         # Python dependencies
├── host.json               # Function app configuration
├── .funcignore            # Files to exclude from deployment
├── local.settings.json.example  # Example configuration
└── README.md              # This file
```

## Additional Considerations

1. **Python Version**: Ensure Python 3.10+ is used (required by Azure Functions v4)
2. **Durable Functions**: Consider using Durable Functions for long-running operations
3. **Application Insights**: Enabled by default for monitoring and diagnostics
4. **Scaling**: Consumption plan auto-scales, but consider dedicated plan for consistent performance
5. **Error Handling**: Function will retry on failure (configurable in host.json)

## Support

For issues or questions, contact the development team.
