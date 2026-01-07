# Azure Automation Runbook to run CMS Pipeline monthly
# This script creates a container instance to run the pipeline

$resourceGroupName = "RGKorteNext"
$location = "eastus"
$registryServer = "kortecmsregistry.azurecr.io"
$registryUsername = "kortecmsregistry"
$registryPassword = "<YOUR_REGISTRY_PASSWORD>"
$imageName = "kortecmsregistry.azurecr.io/cms-pipeline:latest"

# Environment variables (stored as encrypted Automation variables)
$storageConnectionString = Get-AutomationVariable -Name 'AZURE_STORAGE_CONNECTION_STRING'
$googleApiKey = Get-AutomationVariable -Name 'GOOGLE_API_KEY'

# Generate unique container name with timestamp
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$containerName = "cms-pipeline-$timestamp"

Write-Output "Creating container: $containerName"

# Create container instance
New-AzContainerGroup `
    -ResourceGroupName $resourceGroupName `
    -Name $containerName `
    -Image $imageName `
    -OsType Linux `
    -RestartPolicy Never `
    -Cpu 4 `
    -MemoryInGB 8 `
    -Location $location `
    -RegistryCredential (New-AzContainerGroupImageRegistryCredentialObject `
        -Server $registryServer `
        -Username $registryUsername `
        -Password (ConvertTo-SecureString $registryPassword -AsPlainText -Force)) `
    -EnvironmentVariable @(
        (New-AzContainerInstanceEnvironmentVariableObject -Name "AZURE_STORAGE_CONNECTION_STRING" -SecureValue (ConvertTo-SecureString $storageConnectionString -AsPlainText -Force)),
        (New-AzContainerInstanceEnvironmentVariableObject -Name "GOOGLE_API_KEY" -SecureValue (ConvertTo-SecureString $googleApiKey -AsPlainText -Force))
    )

Write-Output "Container $containerName created successfully"
Write-Output "The pipeline will run automatically and outputs will be saved to Azure Blob Storage"
