"""
Azure Function to trigger CMS pipeline container on a schedule
This function creates a container instance to run the CMS pipeline
"""
import logging
import requests
import os
from datetime import datetime
import json


def trigger_container():
    """Create and start a container instance to run the CMS pipeline"""
    
    logging.info(f'CMS Container Trigger executed at: {datetime.utcnow()}')
    
    # Configuration
    subscription_id = "46e2ea4c-f643-4996-bdde-c0da83063906"
    resource_group = "RGKorteNext"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    container_name = f"cms-pipeline-{timestamp}"
    
    # Azure Resource Manager API endpoint
    url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerInstance/containerGroups/{container_name}?api-version=2023-05-01"
    
    try:
        # Get managed identity token
        identity_endpoint = os.environ.get("IDENTITY_ENDPOINT")
        identity_header = os.environ.get("IDENTITY_HEADER")
        
        if not identity_endpoint or not identity_header:
            logging.error("Managed Identity not configured. Enable System-assigned identity for this Function App.")
            return
        
        token_url = f"{identity_endpoint}?resource=https://management.azure.com/&api-version=2019-08-01"
        token_headers = {"X-IDENTITY-HEADER": identity_header}
        
        logging.info("Requesting managed identity token...")
        token_response = requests.get(token_url, headers=token_headers)
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]
        logging.info("Managed identity token obtained successfully")
        
        # Get credentials from environment variables
        storage_conn_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not storage_conn_string or not google_api_key:
            logging.error("Required environment variables not set")
            return
        
        # Container configuration
        container_config = {
            "location": "eastus",
            "properties": {
                "containers": [{
                    "name": "cms-pipeline",
                    "properties": {
                        "image": "kortecmsregistry.azurecr.io/cms-pipeline:latest",
                        "resources": {
                            "requests": {
                                "cpu": 4,
                                "memoryInGb": 8
                            }
                        },
                        "environmentVariables": [
                            {
                                "name": "AZURE_STORAGE_CONNECTION_STRING",
                                "secureValue": storage_conn_string
                            },
                            {
                                "name": "GOOGLE_API_KEY",
                                "secureValue": google_api_key
                            }
                        ]
                    }
                }],
                "osType": "Linux",
                "restartPolicy": "Never",
                "imageRegistryCredentials": [{
                    "server": "kortecmsregistry.azurecr.io",
                    "username": "kortecmsregistry",
                    "password": "<YOUR_REGISTRY_PASSWORD>"
                }]
            }
        }
        
        # Create container
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        logging.info(f"Creating container instance: {container_name}")
        response = requests.put(url, json=container_config, headers=headers, timeout=60)
        
        if response.status_code in [200, 201]:
            logging.info(f"✅ Successfully created container: {container_name}")
            logging.info("Container will begin execution automatically")
            return f"Container {container_name} created successfully"
        else:
            error_msg = f"Failed to create container: {response.status_code} - {response.text}"
            logging.error(error_msg)
            return error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error while creating container: {str(e)}"
        logging.error(error_msg)
        return error_msg
    except KeyError as e:
        error_msg = f"Missing required key in response: {str(e)}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logging.error(error_msg)
        return error_msg
