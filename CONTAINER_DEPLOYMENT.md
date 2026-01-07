# CMS Pipeline - Azure Container Instance Deployment

## ✅ **Current Status: WORKING!**

The Docker container has been successfully built, deployed, and tested. The pipeline completed successfully with all outputs saved to Azure Blob Storage.

---

## **What We Built**

1. **Docker Image**: Contains Python 3.11 with all ML dependencies (pandas, numpy, scikit-learn)
2. **Azure Container Registry**: `kortecmsregistry.azurecr.io`
3. **Container Image**: `kortecmsregistry.azurecr.io/cms-pipeline:latest`
4. **Test Run**: Completed successfully with exit code 0

### **Container Specifications**
- **CPU**: 4 cores
- **Memory**: 8 GB RAM
- **OS**: Linux (amd64)
- **Restart Policy**: Never (runs once and stops)

---

## **How to Run Manually**

To trigger a manual run anytime:

```bash
az container create \
  --name cms-pipeline-$(date +%Y%m%d%H%M%S) \
  --resource-group RGKorteNext \
  --image kortecmsregistry.azurecr.io/cms-pipeline:latest \
  --registry-login-server kortecmsregistry.azurecr.io \
  --registry-username kortecmsregistry \
  --registry-password "<YOUR_REGISTRY_PASSWORD>" \
  --secure-environment-variables \
    AZURE_STORAGE_CONNECTION_STRING="<YOUR_AZURE_STORAGE_CONNECTION_STRING>" \
    GOOGLE_API_KEY="<YOUR_GOOGLE_API_KEY>" \
  --restart-policy Never \
  --os-type Linux \
  --cpu 4 \
  --memory 8 \
  --location eastus
```

### **Check Logs**
```bash
az container logs --name <container-name> --resource-group RGKorteNext
```

### **Check Status**
```bash
az container show --name <container-name> --resource-group RGKorteNext --query "containers[0].instanceView.currentState"
```

---

## **Set Up Monthly Schedule (Option 1: Azure Automation)**

### **Step 1: Create Automation Account**
```bash
az automation account create \
  --name cms-automation \
  --resource-group RGKorteNext \
  --location eastus
```

### **Step 2: Import PowerShell Runbook**
1. Go to Azure Portal → Automation Accounts → cms-automation
2. Click "Runbooks" → "Create a runbook"
3. Name: `Run-CMS-Pipeline`
4. Type: PowerShell
5. Paste contents from `run-pipeline.ps1`
6. Click "Publish"

### **Step 3: Create Encrypted Variables**
1. Go to "Variables" in the Automation Account
2. Create two encrypted variables:
   - Name: `AZURE_STORAGE_CONNECTION_STRING`
     Value: `<YOUR_AZURE_STORAGE_CONNECTION_STRING>`
   - Name: `GOOGLE_API_KEY`
     Value: `<YOUR_GOOGLE_API_KEY>`

### **Step 4: Schedule the Runbook**
1. Go to the runbook → "Schedules" → "Add a schedule"
2. Create new schedule:
   - Name: Monthly CMS Pipeline
   - Starts: First of next month at 3:00 AM CST
   - Recurrence: Monthly
   - On these days: 1
3. Link to runbook

---

## **Set Up Monthly Schedule (Option 2: Azure Logic Apps - GUI Method)**

### **Easiest Method - Use Azure Portal:**

1. **Create Logic App**:
   - Go to Azure Portal → Create Resource → Logic App
   - Name: `cms-pipeline-scheduler`
   - Resource Group: `RGKorteNext`
   - Region: East US
   - Plan: Consumption

2. **Design the Workflow**:
   - Open Logic Apps Designer
   - Add trigger: "Recurrence"
     - Frequency: Month
     - Interval: 1
     - Time zone: Central Standard Time
     - On these days: 1
     - At these hours: 3
     - At these minutes: 0
   
   - Add action: "HTTP"
     - Method: PUT
     - URI: `https://management.azure.com/subscriptions/<YOUR-SUBSCRIPTION-ID>/resourceGroups/RGKorteNext/providers/Microsoft.ContainerInstance/containerGroups/cms-pipeline-@{utcNow('yyyyMMddHHmmss')}?api-version=2023-05-01`
     - Authentication: Managed Identity
     - Body: (Copy the container configuration JSON)

3. **Enable Managed Identity**:
   - Go to Logic App → Identity → System assigned → On
   - Copy the Object ID

4. **Grant Permissions**:
   ```bash
   az role assignment create \
     --assignee <OBJECT-ID> \
     --role Contributor \
     --scope /subscriptions/<SUBSCRIPTION-ID>/resourceGroups/RGKorteNext
   ```

---

## **Update the Code**

If you need to update the pipeline code:

```bash
# 1. Edit master.py
# 2. Rebuild and push
cd /Users/brandonadmin/AzureProjects/CMS/CMSTimerIncremental
docker build --platform linux/amd64 -t cms-pipeline:latest .
docker tag cms-pipeline:latest kortecmsregistry.azurecr.io/cms-pipeline:latest
docker push kortecmsregistry.azurecr.io/cms-pipeline:latest

# 3. Next scheduled run will use the updated image automatically
```

---

## **Cost Estimate**

- **Container Registry (Basic)**: ~$5/month
- **Container Instances**: ~$0.30-0.50 per execution (4 vCPU, 8GB RAM, ~3 min runtime)
- **Monthly Total**: ~$5.50-6.00/month

---

## **Outputs**

The pipeline saves these files to the `cmsfiles` container in the `cmsdata` storage account:

- `cms_master.csv` - Complete historical dataset
- `cms_master_MM_YYYY.csv` - Timestamped snapshot
- `cms_dedupe.csv` - Most recent two years, deduped
- `cms_visuals.csv` - Final output with ML predictions
- `geocode_data.csv` - Hospital geocode data
- `inter_hospital_distances.csv` - Distance calculations
- `HOSP10_YYYY_*.csv` - Raw HCRIS files

---

## **Troubleshooting**

### **Out of Memory Error (Exit Code 137)**
- Increase memory in the container create command: `--memory 8` or higher

### **Container Won't Start**
- Check registry credentials
- Verify environment variables are set correctly
- Check logs: `az container logs --name <name> --resource-group RGKorteNext`

### **Pipeline Errors**
- Check container logs for Python errors
- Verify Azure Storage connection string is valid
- Verify Google API key is valid (if geocoding needed)

---

## **Next Steps**

**Recommended**: Set up Option 2 (Logic Apps via Portal) - it's the easiest to configure and monitor through the GUI.

Would you like help setting up either scheduling option?
