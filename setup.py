#!/usr/bin/env python3
"""
Configuration and setup script for Azure deployment.
Provides utilities for environment setup and validation.
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Optional, Tuple


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def install_system_dependencies() -> bool:
    """Install system-level dependencies for solvers."""
    print("Installing system dependencies...")
    
    # Commands to install solver dependencies
    commands = []
    
    # Detect OS and add appropriate commands
    if sys.platform.startswith('linux'):
        # Ubuntu/Debian commands
        commands.extend([
            'apt-get update',
            'apt-get install -y build-essential',
            'apt-get install -y libglpk-dev glpk-utils',  # GLPK
            'apt-get install -y coinor-cbc',  # CBC
        ])
    elif sys.platform == 'darwin':
        # macOS with Homebrew
        commands.extend([
            'brew install glpk',
            'brew install cbc',
        ])
    else:
        print("Warning: Automatic system dependency installation not supported on this OS")
        print("Please install GLPK and CBC manually")
        return True
    
    success = True
    for cmd in commands:
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            print(f"✓ Command completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed: {e}")
            print(f"Error output: {e.stderr}")
            success = False
        except FileNotFoundError:
            print(f"✗ Command not found: {cmd.split()[0]}")
            success = False
    
    return success


def check_data_files(data_path: str = "test_data/") -> Tuple[bool, List[str]]:
    """Check if required data files exist."""
    required_files = [
        "config.json",
        "crude_availability.csv",
        "crudes_info.csv", 
        "products_info.csv",
        "time_of_travel.csv"
    ]
    
    missing_files = []
    
    print(f"Checking data files in {data_path}...")
    
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files


def validate_config_file(config_path: str = "test_data/config.json") -> bool:
    """Validate the configuration file format."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = [
            "DAYS", "INVENTORY_MAX_VOLUME", "MaxTransitions",
            "demurrage_cost", "vessel_max_limit", "default_capacity",
            "turn_down_capacity", "two_parcel_vessel_capacity", 
            "three_parcel_vessel_capacity", "solver_time_limit_seconds",
            "schedule_month", "schedule_year", "plant_capacity_reduction_window"
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"✗ Config validation failed. Missing keys: {missing_keys}")
            return False
        else:
            print("✓ Configuration file valid")
            return True
            
    except json.JSONDecodeError as e:
        print(f"✗ Config file JSON error: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        return False


def create_azure_deployment_files():
    """Create Azure-specific deployment files."""
    
    # Create Dockerfile
    dockerfile_content = """
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libglpk-dev \\
    glpk-utils \\
    coinor-cbc \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY test_data/ ./test_data/

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py", "--help"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Create Azure Container Instances deployment template
    aci_template = {
        "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
        "contentVersion": "1.0.0.0",
        "parameters": {
            "containerGroupName": {
                "type": "string",
                "defaultValue": "aegis-optimizer",
                "metadata": {
                    "description": "Name for the container group"
                }
            },
            "vesselCount": {
                "type": "int",
                "defaultValue": 6,
                "metadata": {
                    "description": "Number of vessels to optimize"
                }
            },
            "timeLimit": {
                "type": "int", 
                "defaultValue": 7200,
                "metadata": {
                    "description": "Solver time limit in seconds"
                }
            }
        },
        "variables": {},
        "resources": [
            {
                "type": "Microsoft.ContainerInstance/containerGroups",
                "apiVersion": "2021-09-01",
                "name": "[parameters('containerGroupName')]",
                "location": "[resourceGroup().location]",
                "properties": {
                    "containers": [
                        {
                            "name": "aegis-optimizer",
                            "properties": {
                                "image": "your-registry/aegis-scheduler:latest",
                                "resources": {
                                    "requests": {
                                        "cpu": 4,
                                        "memoryInGb": 8
                                    }
                                },
                                "command": [
                                    "python", "main.py",
                                    "--vessel-count", "[parameters('vesselCount')]",
                                    "--time-limit", "[parameters('timeLimit')]",
                                    "--optimization-type", "throughput",
                                    "--output-dir", "/app/results"
                                ],
                                "environmentVariables": [
                                    {
                                        "name": "PYTHONUNBUFFERED",
                                        "value": "1"
                                    }
                                ]
                            }
                        }
                    ],
                    "osType": "Linux",
                    "restartPolicy": "Never"
                }
            }
        ],
        "outputs": {
            "containerIPv4Address": {
                "type": "string",
                "value": "[reference(resourceId('Microsoft.ContainerInstance/containerGroups', parameters('containerGroupName'))).ipAddress.ip]"
            }
        }
    }
    
    with open("azure-aci-template.json", "w") as f:
        json.dump(aci_template, f, indent=2)
    
    # Create deployment script
    deploy_script = """#!/bin/bash
# Azure deployment script for Aegis Scheduler

set -e

echo "Aegis Scheduler Azure Deployment"
echo "================================="

# Configuration
RESOURCE_GROUP="aegis-optimization-rg"
LOCATION="eastus"
CONTAINER_REGISTRY="aegisregistry"
IMAGE_NAME="aegis-scheduler"
IMAGE_TAG="latest"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI not found. Please install Azure CLI first."
    exit 1
fi

# Login to Azure (if not already logged in)
echo "Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
fi

# Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create container registry
echo "Creating container registry..."
az acr create --resource-group $RESOURCE_GROUP --name $CONTAINER_REGISTRY --sku Basic

# Build and push Docker image
echo "Building Docker image..."
az acr build --registry $CONTAINER_REGISTRY --image $IMAGE_NAME:$IMAGE_TAG .

# Deploy container instance
echo "Deploying container instance..."
az deployment group create \\
    --resource-group $RESOURCE_GROUP \\
    --template-file azure-aci-template.json \\
    --parameters containerGroupName=aegis-optimizer-$(date +%s)

echo "Deployment completed!"
echo "Check the Azure portal for container status and logs."
"""
    
    with open("deploy-azure.sh", "w") as f:
        f.write(deploy_script)
    
    # Make deploy script executable
    os.chmod("deploy-azure.sh", 0o755)
    
    print("✓ Azure deployment files created:")
    print("  - Dockerfile")
    print("  - azure-aci-template.json")
    print("  - deploy-azure.sh")


def run_system_check() -> bool:
    """Run comprehensive system check."""
    print("AEGIS SCHEDULER - SYSTEM CHECK")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    print()
    
    # Check data files
    data_ok, missing_files = check_data_files()
    if not data_ok:
        all_checks_passed = False
        print(f"Missing data files: {missing_files}")
    
    print()
    
    # Validate config
    if not validate_config_file():
        all_checks_passed = False
    
    print()
    
    # Check solvers
    try:
        from solver_manager import get_available_solvers
        solvers = get_available_solvers()
        available_solvers = [s for s, avail in solvers.items() if avail]
        
        if available_solvers:
            print(f"✓ Available solvers: {', '.join(available_solvers)}")
        else:
            print("✗ No solvers available")
            all_checks_passed = False
    except ImportError as e:
        print(f"✗ Could not check solvers: {e}")
        all_checks_passed = False
    
    print()
    print("=" * 50)
    
    if all_checks_passed:
        print("✓ All system checks passed!")
        print("System is ready for optimization.")
    else:
        print("✗ Some system checks failed.")
        print("Please resolve the issues above before running optimization.")
    
    return all_checks_passed


def main():
    """Main configuration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aegis Scheduler Configuration & Setup")
    parser.add_argument('--check', action='store_true', help='Run system check')
    parser.add_argument('--install-deps', action='store_true', help='Install system dependencies')
    parser.add_argument('--create-azure-files', action='store_true', help='Create Azure deployment files')
    parser.add_argument('--all', action='store_true', help='Run all setup tasks')
    
    args = parser.parse_args()
    
    if args.all:
        run_system_check()
        print()
        install_system_dependencies()
        print()
        create_azure_deployment_files()
    elif args.check:
        run_system_check()
    elif args.install_deps:
        install_system_dependencies()
    elif args.create_azure_files:
        create_azure_deployment_files()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()