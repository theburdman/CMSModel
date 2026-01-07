# Azure Function Deployment Guide

## Pre-Deployment Checklist

- [ ] Azure subscription is active
- [ ] Azure CLI installed and authenticated (`az login`)
- [ ] Azure Functions Core Tools v4 installed
- [ ] Python 3.10 or higher installed
- [ ] Storage account `cmsdata` exists in resource group `RGKorteNext`
- [ ] Google Maps API key obtained
- [ ] Connection string for `cmsdata` storage account available

## Step 1: Install Azure Functions Core Tools (if needed)

```bash
# macOS
brew tap azure/functions
brew install azure-functions-core-tools@4

# Windows (via Chocolatey)
choco install azure-functions-core-tools-4

# Linux (Ubuntu/Debian)
wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install azure-functions-core-tools-4
```

## Step 2: Verify Project Structure

Your project should have these files:
- `function_app.py` ✓
- `master.py` ✓
- `requirements.txt` ✓
- `host.json` ✓
- `.funcignore` ✓

## Step 3: Create the Function App in Azure

Run these commands from your project directory:

```bash
# Login to Azure
az login

# Set your subscription (if you have multiple)
az account set --subscription "YOUR_SUBSCRIPTION_NAME_OR_ID"

# Verify resource group exists
az group show --name RGKorteNext

# Create the Function App
az functionapp create \
  --resource-group RGKorteNext \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.10 \
  --functions-version 4 \
  --name korteCMSFunction \
  --storage-account cmsdata \
  --os-type Linux
```

**Note**: If the function app already exists, you'll see an error. That's okay - proceed to deployment.

## Step 4: Configure Application Settings

Add your secrets to the Function App:

```bash
# Add Azure Storage connection string
az functionapp config appsettings set \
  --name korteCMSFunction \
  --resource-group RGKorteNext \
  --settings "AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=cmsdata;AccountKey=YOUR_KEY_HERE;EndpointSuffix=core.windows.net"

# Add Google API key
az functionapp config appsettings set \
  --name korteCMSFunction \
  --resource-group RGKorteNext \
  --settings "GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY"
```

**Or configure via Azure Portal:**
1. Navigate to: Azure Portal → RGKorteNext → korteCMSFunction
2. Click **Configuration** → **Application settings**
3. Add:
   - `AZURE_STORAGE_CONNECTION_STRING`
   - `GOOGLE_API_KEY`
4. Click **Save**

## Step 5: Deploy the Function

From your project directory:

```bash
func azure functionapp publish korteCMSFunction
```

This will:
- Package your code
- Upload to Azure
- Install Python dependencies
- Deploy the function

## Step 6: Verify Deployment

```bash
# Check function app status
az functionapp show \
  --name korteCMSFunction \
  --resource-group RGKorteNext \
  --query "state"

# List functions in the app
az functionapp function show \
  --name korteCMSFunction \
  --resource-group RGKorteNext \
  --function-name korteCMSFunction
```

## Step 7: Configure for Testing

For initial testing, you may want to run the function more frequently:

1. **Option A: Update schedule in code** (before deployment):
   Edit `function_app.py`:
   ```python
   @app.schedule(
       schedule="0 */10 * * * *",  # Every 10 minutes for testing
       ...
   )
   ```

2. **Option B: Use manual HTTP trigger**:
   Get the function URL:
   ```bash
   az functionapp function keys list \
     --name korteCMSFunction \
     --resource-group RGKorteNext \
     --function-name cms_manual_trigger \
     --query "default" \
     --output tsv
   ```
   
   Then trigger manually:
   ```bash
   curl -X POST "https://kortecmsfunction.azurewebsites.net/api/cms_manual_trigger?code=YOUR_FUNCTION_KEY"
   ```

## Step 8: Monitor Execution

### View Live Logs:
```bash
func azure functionapp logstream korteCMSFunction
```

### Or in Azure Portal:
1. Navigate to: Azure Portal → RGKorteNext → korteCMSFunction
2. Click **Log stream** (in the left menu)
3. Watch real-time logs

### Check Execution History:
1. Navigate to: Azure Portal → RGKorteNext → korteCMSFunction
2. Click **Functions** → **korteCMSFunction** → **Monitor**
3. View invocation history and success/failure status

## Step 9: Verify Blob Storage Outputs

After a successful run, check your blob storage:

```bash
# List blobs in the container
az storage blob list \
  --account-name cmsdata \
  --container-name cmsfiles \
  --output table
```

Expected files:
- `cms_master.csv`
- `cms_dedupe.csv`
- `cms_visuals.csv`
- `geocode_data.csv`
- `inter_hospital_distances.csv`

## Step 10: Switch to Production Schedule

Once testing is complete, update the schedule to monthly:

1. Edit `function_app.py`:
   ```python
   @app.schedule(
       schedule="0 0 18 1 * *",  # 6 PM on the 1st of each month
       ...
   )
   ```

2. Redeploy:
   ```bash
   func azure functionapp publish korteCMSFunction
   ```

## Troubleshooting

### Function fails to start:
- Check Application Settings are configured correctly
- Verify Python runtime version in Azure matches requirements
- Review deployment logs: `az functionapp deployment source show-logs`

### Function times out:
- Check `host.json` for `functionTimeout` setting (default: 60 minutes)
- Consider Premium plan for longer timeouts
- Check Azure Portal → Function → Monitor for execution duration

### Can't see logs:
- Ensure Application Insights is enabled (should be automatic)
- Check Log Analytics workspace is connected
- Try Log Stream in portal

### Deployment fails:
- Ensure `requirements.txt` doesn't have conflicting versions
- Check that all files are included (not in `.funcignore`)
- Try local build first: `func start`

### Secrets not working:
- Verify Application Settings names match code exactly
- Check for extra spaces in Azure Portal settings
- Restart the Function App after changing settings

## Post-Deployment Monitoring

### Set up Alerts:
1. Azure Portal → korteCMSFunction → Alerts
2. Create alert rules for:
   - Function failures
   - Execution duration > 50 minutes
   - HTTP 5xx errors

### Regular Checks:
- Monthly: Review execution logs
- Quarterly: Verify output data quality
- As needed: Update API keys if they expire

## Useful Commands Reference

```bash
# View function app details
az functionapp show --name korteCMSFunction --resource-group RGKorteNext

# List application settings
az functionapp config appsettings list --name korteCMSFunction --resource-group RGKorteNext

# Restart function app
az functionapp restart --name korteCMSFunction --resource-group RGKorteNext

# Stop function app
az functionapp stop --name korteCMSFunction --resource-group RGKorteNext

# Start function app
az functionapp start --name korteCMSFunction --resource-group RGKorteNext

# Get function app URL
az functionapp show --name korteCMSFunction --resource-group RGKorteNext --query "defaultHostName" -o tsv

# Delete function app (use with caution!)
az functionapp delete --name korteCMSFunction --resource-group RGKorteNext
```

## Next Steps After Deployment

1. **Initial Test Run**: Trigger manually to verify everything works
2. **Data Validation**: Check blob storage for correct outputs
3. **Schedule Optimization**: Adjust timing based on CMS data release schedule
4. **Cost Monitoring**: Track consumption plan usage in Azure Portal
5. **Documentation**: Update team on new automated process

## Support Resources

- Azure Functions Documentation: https://docs.microsoft.com/azure/azure-functions/
- Azure Functions Python Guide: https://docs.microsoft.com/azure/azure-functions/functions-reference-python
- Azure CLI Reference: https://docs.microsoft.com/cli/azure/functionapp

---

**Deployment completed?** ✓ Remember to update the schedule to production settings after successful testing!
