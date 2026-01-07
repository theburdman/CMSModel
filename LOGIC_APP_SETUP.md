# Logic App Scheduler - Setup Complete

**Last Updated**: December 17, 2025

---

## ✅ COMPLETED SETUP

The Logic App `cms-pipeline-scheduler` is fully configured and deployed:

- **Status**: Enabled and running
- **Location**: East US
- **Resource Group**: RGKorteNext
- **Managed Identity**: Enabled (Principal ID: `879133ab-133a-4535-96d9-04ba3ea7c4d2`)

### Schedule Configuration
- **Trigger**: 1st of every month
- **Time**: 3:00 AM Central Standard Time
- **Recurrence**: Monthly on day 1

### Container Configuration
- **Image**: `kortecmsregistry.azurecr.io/cms-pipeline:latest`
- **CPU**: 4 cores
- **Memory**: 8 GB RAM
- **Restart Policy**: Never (runs once and exits)
- **Environment Variables**: Azure Storage + Google API Key (configured securely)

---

## 🔴 REQUIRED: IT Permission

The Logic App is ready but **cannot create containers yet** because it lacks the necessary permissions.

### Command for IT to Run:

```bash
az role assignment create \
  --assignee 879133ab-133a-4535-96d9-04ba3ea7c4d2 \
  --role Contributor \
  --scope /subscriptions/46e2ea4c-f643-4996-bdde-c0da83063906/resourceGroups/RGKorteNext
```

### Or via Azure Portal:

1. Navigate to: **Resource Groups** → **RGKorteNext** → **Access control (IAM)**
2. Click **Add role assignment**
3. Select role: **Contributor**
4. Assign to: **Managed identity** → **Logic App**
5. Select: **cms-pipeline-scheduler**
6. Click **Save**

**Why this is needed**: The Logic App uses the Azure Management API to create container instances. The Contributor role allows it to create and manage containers within the RGKorteNext resource group.

---

## 🧪 TESTING (After Permissions Granted)

### Test the Logic App Manually

Once IT grants the permissions, you can test it immediately without waiting for the schedule:

```bash
az logic workflow run trigger --name cms-pipeline-scheduler --resource-group RGKorteNext --trigger-name Monthly_Schedule
```

### Check Run History

View the Logic App execution history:

```bash
az logic workflow show --name cms-pipeline-scheduler --resource-group RGKorteNext --query "definition.triggers.Monthly_Schedule.evaluatedRecurrence"
```

Or in the **Azure Portal**:
1. Go to **Logic Apps** → **cms-pipeline-scheduler**
2. Click **Overview** → **Runs history**
3. View execution details, inputs/outputs, and any errors

### Verify Container Created

After a successful run, check for the new container:

```bash
az container list --resource-group RGKorteNext --output table
```

The container name will be: `cms-pipeline-YYYYMMDDHHMMSS`

### Check Container Logs

```bash
az container logs --name <container-name> --resource-group RGKorteNext
```

---

## 📊 HOW IT WORKS

1. **Trigger**: On the 1st of each month at 3 AM CST, the Logic App wakes up
2. **API Call**: Makes an HTTP PUT request to Azure Management API
3. **Container Creation**: Creates a new Azure Container Instance with:
   - Your Docker image from Azure Container Registry
   - Environment variables for Azure Storage and Google API
   - 4 CPU cores and 8GB RAM
4. **Execution**: Container runs `master.py`, processes data, saves to blob storage
5. **Cleanup**: Container exits and stops (restart policy = Never)
6. **Outputs**: All files are saved to `cmsdata/cmsfiles` blob storage

---

## 🎯 NEXT STEPS

1. **Send permission request to IT** (use the command/portal instructions above)
2. **Test the Logic App** once permissions are granted
3. **Verify outputs** in Azure Blob Storage after test run
4. **Monitor first scheduled run** on January 1st, 2026 at 3 AM CST

---

## 💰 COST ESTIMATE

- **Logic App**: Free (Consumption plan, 1 execution/month)
- **Container Instance**: ~$0.30-0.50 per run
- **Container Registry**: ~$5/month (already in place)
- **Monthly Total**: ~$5.50-6.00/month

---

## 🔄 UPDATING THE CODE

If you need to update the pipeline code in the future:

```bash
cd /Users/brandonadmin/AzureProjects/CMS/CMSTimerIncremental

# Edit master.py as needed

# Rebuild and push Docker image
docker build --platform linux/amd64 -t cms-pipeline:latest .
docker tag cms-pipeline:latest kortecmsregistry.azurecr.io/cms-pipeline:latest
docker push kortecmsregistry.azurecr.io/cms-pipeline:latest
```

The Logic App will automatically use the updated image on the next run.

---

## 📋 COMPARISON: Logic App vs Azure Function

You chose the Logic App - here's why it's better for this use case:

| Feature | Logic App | Azure Function |
|---------|-----------|----------------|
| **Setup** | ✅ Visual designer, easy config | ⚠️ Code-based, more complex |
| **Monitoring** | ✅ Built-in GUI, run history | ⚠️ Requires Application Insights |
| **Permissions** | ✅ Single managed identity | ✅ Same (managed identity) |
| **Scheduling** | ✅ Native recurrence trigger | ✅ Timer trigger |
| **Cost** | ✅ Free for 1 run/month | ✅ Similar (minimal) |
| **Maintenance** | ✅ No code to deploy | ⚠️ Need to deploy function code |

**Winner**: Logic App - simpler, easier to monitor, no code to maintain!

---

## ✅ READY TO GO

Everything is configured. Once IT grants the permission, the Logic App will:
- Run automatically on the 1st of every month at 3 AM CST
- Create a container instance with your pipeline
- Process the CMS data
- Save outputs to Azure Blob Storage
- Clean up after itself

**No manual intervention needed!**
