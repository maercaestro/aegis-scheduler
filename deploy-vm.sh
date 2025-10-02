#!/bin/bash
# Azure VM deployment script for Aegis Scheduler

echo "🚀 Aegis Scheduler - Azure VM Deployment"
echo "========================================"

# Configuration (modify these as needed)
RESOURCE_GROUP="aegis-optimization-rg"
LOCATION="eastus"
VM_NAME="aegis-vm"
VM_SIZE="Standard_D4s_v3"  # 4 vCPUs, 16GB RAM - fits student quota
VM_IMAGE="Ubuntu2204"
ADMIN_USERNAME="azureuser"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Error: Azure CLI not found. Please install Azure CLI first."
    echo "   Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

echo "✅ Azure CLI found"

# Login to Azure (if not already logged in)
echo "🔐 Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
else
    echo "✅ Already logged in to Azure"
fi

# Show current subscription
SUBSCRIPTION=$(az account show --query name -o tsv)
echo "📋 Current subscription: $SUBSCRIPTION"

# Create resource group
echo "📁 Creating resource group: $RESOURCE_GROUP..."
az group create --name $RESOURCE_GROUP --location $LOCATION
if [ $? -eq 0 ]; then
    echo "✅ Resource group created successfully"
else
    echo "❌ Failed to create resource group"
    exit 1
fi

# Generate SSH key if it doesn't exist
SSH_KEY_PATH="$HOME/.ssh/aegis_vm_key"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "🔑 Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "aegis-vm-key"
    echo "✅ SSH key generated: $SSH_KEY_PATH"
else
    echo "✅ SSH key already exists: $SSH_KEY_PATH"
fi

# Create VM
echo "💻 Creating virtual machine: $VM_NAME..."
echo "   Size: $VM_SIZE (4 vCPUs, 16GB RAM - Student quota friendly)"
echo "   Image: $VM_IMAGE"
echo "   This may take 3-5 minutes..."

az vm create \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --image $VM_IMAGE \
    --size $VM_SIZE \
    --admin-username $ADMIN_USERNAME \
    --ssh-key-values "${SSH_KEY_PATH}.pub" \
    --public-ip-sku Standard \
    --storage-sku Premium_LRS

if [ $? -eq 0 ]; then
    echo "✅ Virtual machine created successfully"
else
    echo "❌ Failed to create virtual machine"
    exit 1
fi

# Get VM public IP
echo "🌐 Getting VM public IP..."
VM_IP=$(az vm show -d -g $RESOURCE_GROUP -n $VM_NAME --query publicIps -o tsv)
echo "✅ VM Public IP: $VM_IP"

# Open necessary ports (SSH is already open by default)
echo "🔓 Opening ports for optimization..."
az vm open-port --resource-group $RESOURCE_GROUP --name $VM_NAME --port 22 --priority 1000

echo ""
echo "🎉 VM Deployment Complete!"
echo "=========================="
echo "VM Name: $VM_NAME"
echo "Public IP: $VM_IP"
echo "Username: $ADMIN_USERNAME"
echo "SSH Key: $SSH_KEY_PATH"
echo ""
echo "Connect to your VM:"
echo "ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
echo ""
echo "Next steps:"
echo "1. Connect to the VM using the SSH command above"
echo "2. Run the setup script: ./setup-vm.sh"
echo "3. Start optimization: python main.py --vessel-count 6 --time-limit 28800"