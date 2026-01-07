# CMS Pipeline Deployment Status

**Last Updated**: December 13, 2025

---

## ✅ WHAT'S WORKING

### **Docker Container (SUCCESSFUL)**
- **Image**: `kortecmsregistry.azurecr.io/cms-pipeline:latest`
- **Registry**: `kortecmsregistry` (Azure Container Registry)
- **Test Status**: ✅ Successfully ran end-to-end (exit code 0)
- **Resources**: 4 CPU, 8GB RAM
- **All outputs saved** to Azure Blob Storage

### **Manual Run Command** (Ready to Use)
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

---

## ⏳ PENDING (Waiting on IT Permissions)

### **Option 1: Logic App Scheduler** (Created, needs permission)
- **Name**: `cms-pipeline-scheduler`
- **Schedule**: 1st of month at 3 AM CST
- **Status**: Created, managed identity enabled
- **Needs**: Contributor role on RGKorteNext
- **Principal ID**: `879133ab-133a-4535-96d9-04ba3ea7c4d2`

**Permission Command for IT:**
```bash
az role assignment create \
  --assignee 879133ab-133a-4535-96d9-04ba3ea7c4d2 \
  --role Contributor \
  --scope /subscriptions/46e2ea4c-f643-4996-bdde-c0da83063906/resourceGroups/RGKorteNext
```

### **Option 2: Azure Function Scheduler** (Deployed, needs permission)
- **Name**: `korteCMSFunction` (container_scheduler function)
- **Schedule**: 1st of month at 3 AM CST (9 AM UTC)
- **Status**: Deployed (lightweight, only 3.33 KB)
- **Needs**: Contributor role on RGKorteNext
- **Principal ID**: `8429c132-01df-4433-85c0-f2747a6a8b39`

**Permission Command for IT:**
```bash
az role assignment create \
  --assignee 8429c132-01df-4433-85c0-f2747a6a8b39 \
  --role Contributor \
  --scope /subscriptions/46e2ea4c-f643-4996-bdde-c0da83063906/resourceGroups/RGKorteNext
```

**Or via Azure Portal:**
1. Resource Groups → `RGKorteNext` → Access control (IAM)
2. Add role assignment → Contributor
3. Assign to: Managed identity
4. Select: Logic App `cms-pipeline-scheduler` OR Function App `korteCMSFunction`

---

## 📁 IMPORTANT FILES

### **Deployment Files**
- `Dockerfile` - Container image definition
- `CONTAINER_DEPLOYMENT.md` - Full deployment guide
- `DEPLOYMENT_STATUS.md` - This file (quick reference)

### **Code Files**
- `master.py` - Main pipeline (runs in container)
- `master_local.py` - Local version (for testing)
- `container_trigger.py` - Function code to trigger containers
- `function_app.py` - Azure Functions entry point

### **Configuration**
- Container Registry: `kortecmsregistry.azurecr.io`
- Storage Account: `cmsdata`
- Blob Container: `cmsfiles`
- Resource Group: `RGKorteNext`

---

## 🔍 WHY WE SWITCHED TO CONTAINERS

**Azure Functions Failed Because:**
- ML packages (pandas, numpy, scikit-learn) = 1.5+ GB
- Build environment disk space exhausted during remote build
- "No space left on device" error during ZIP creation
- Even Premium plan has limited build disk space

**Container Solution:**
- ✅ Pre-built image with all dependencies
- ✅ No build size limits
- ✅ Successfully tested and working
- ✅ Can be triggered on schedule via Logic App or Function

---

## 🎯 NEXT STEPS

1. **For Manual Runs**: Use the container create command above
2. **For Scheduling**: Ask IT to grant one of the permissions above
3. **To Update Code**: Rebuild Docker image and push (see CONTAINER_DEPLOYMENT.md)

---

## 📊 OUTPUTS (in Azure Blob Storage)

All files saved to: `cmsdata` storage account → `cmsfiles` container

- `cms_master.csv` - Complete historical dataset
- `cms_master_MM_YYYY.csv` - Timestamped snapshot
- `cms_dedupe.csv` - Most recent data (deduped)
- `cms_visuals.csv` - Final output with ML predictions
- `geocode_data.csv` - Hospital coordinates
- `inter_hospital_distances.csv` - Distance calculations
- `HOSP10_YYYY_*.csv` - Raw HCRIS files

---

## 💰 COST ESTIMATE

- **Container Registry**: ~$5/month
- **Container Execution**: ~$0.30-0.50 per run
- **Monthly Total**: ~$5.50-6.00/month
