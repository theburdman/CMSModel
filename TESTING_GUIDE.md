# Testing Guide - Before Deployment

## ✅ Step 1: Check Syntax and Indentation

Run this command to verify your code has no syntax errors:

```bash
python3 check_syntax.py
```

**Expected output:**
```
============================================================
Python Syntax & Indentation Checker
============================================================
...
✓ ALL CHECKS PASSED
============================================================
Your code is ready to deploy!
```

If you see errors, they will show the file and line number. Fix them before proceeding.

---

## ✅ Step 2: (Optional) Test Pipeline Locally

If you want to verify the pipeline works end-to-end before deploying:

### Setup Environment Variables

**Option A: Temporary (for this session only)**
```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=cmsdata;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net"
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

**Option B: Use the hardcoded values in master.py**
The code already has fallback values, so if environment variables aren't set, it will use those.

### Run Local Test

```bash
python3 test_local.py
```

**What this does:**
- Connects to Azure Blob Storage
- Downloads latest HCRIS data
- Processes all data
- Runs geocoding (if needed)
- Calculates distances
- Runs ML predictions
- Saves all outputs to blob storage

**Expected outcome:**
```
============================================================
✓ SUCCESS: Pipeline completed without errors
============================================================

Check blob storage for outputs:
  - cms_master.csv
  - cms_dedupe.csv
  - cms_visuals.csv
  - geocode_data.csv
  - inter_hospital_distances.csv
```

**Note:** This test will:
- Take 10-60 minutes depending on data size
- Use your actual Azure storage
- Make real API calls to Google (if geocoding is needed)
- Create/modify files in your blob storage

---

## ✅ Step 3: Deploy to Azure

Once tests pass, you're ready to deploy!

See `DEPLOYMENT_GUIDE.md` for complete deployment instructions.

---

## Troubleshooting

### Syntax Checker Fails

**Problem:** `TabError: inconsistent use of tabs and spaces`
**Solution:** Your editor mixed tabs and spaces. Python requires consistent indentation.

**Fix:**
1. Open the file in your editor
2. Convert all indentation to tabs (or spaces, but be consistent)
3. Most editors have "Convert Indentation to Tabs" feature
4. Run `python3 check_syntax.py` again

### Local Test Fails

**Problem:** `Error reading [filename]: ResourceNotFoundError`
**Solution:** File doesn't exist in blob storage yet.

**Fix:** This is normal for first run. The pipeline will create the file.

**Problem:** `ConnectionError` or `AuthenticationError`
**Solution:** Check your connection string and API keys

**Fix:**
1. Verify `AZURE_STORAGE_CONNECTION_STRING` is correct
2. Verify `GOOGLE_API_KEY` is valid
3. Check network connection

**Problem:** Function times out or takes too long
**Solution:** This is expected for first run or with large datasets

**Options:**
- Wait it out (can take 60+ minutes)
- Skip local testing and deploy directly
- Test on a smaller dataset first

---

## Summary

### Minimum Testing (Required)
```bash
python3 check_syntax.py
```
This ensures no syntax errors. Takes 5 seconds.

### Full Testing (Optional but Recommended)
```bash
python3 check_syntax.py && python3 test_local.py
```
This runs the full pipeline locally. Takes 10-60 minutes.

### Skip Local Testing
If you prefer, you can skip local testing and deploy directly to Azure. The Azure Function will log any errors, and you can debug from there.

---

## What's Next?

After successful testing:

1. **Deploy to Azure** (see `DEPLOYMENT_GUIDE.md`)
2. **Configure secrets** in Azure Portal
3. **Test the deployed function** with manual trigger
4. **Monitor execution** in Azure Portal

Good luck! 🚀
