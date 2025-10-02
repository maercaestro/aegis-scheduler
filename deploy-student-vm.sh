#!/bin/bash
# Azure VM deployment script for Aegis Scheduler - Student Subscription Version

echo "🎓 Aegis Scheduler - Azure Student VM Deployment"
echo "==============================================="

# Configuration optimized for Azure Student subscription
RESOURCE_GROUP="aegis-student-rg"
LOCATION="eastus"
VM_NAME="aegis-student-vm"
VM_SIZE="Standard_B4ms"  # 4 vCPUs, 16GB RAM - student quota friendly
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

# Show current subscription and check if it's student
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
SSH_KEY_PATH="$HOME/.ssh/aegis_student_key"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "🔑 Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "aegis-student-vm-key"
    echo "✅ SSH key generated: $SSH_KEY_PATH"
else
    echo "✅ SSH key already exists: $SSH_KEY_PATH"
fi

# Check quota before creating VM
echo "📊 Checking quota limits..."
CURRENT_CORES=$(az vm list-usage --location $LOCATION --query "[?name.value=='cores'].currentValue" -o tsv 2>/dev/null || echo "0")
CORE_LIMIT=$(az vm list-usage --location $LOCATION --query "[?name.value=='cores'].limit" -o tsv 2>/dev/null || echo "6")
echo "   Current cores used: $CURRENT_CORES"
echo "   Core limit: $CORE_LIMIT"
echo "   Cores needed: 4"

if [ $((CURRENT_CORES + 4)) -gt $CORE_LIMIT ]; then
    echo "⚠️  Warning: This deployment may exceed your quota limits."
    echo "   Consider using Standard_B2ms (2 vCPUs, 8GB RAM) instead."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled. Try with smaller VM size."
        exit 1
    fi
fi

# Create VM
echo "💻 Creating virtual machine: $VM_NAME..."
echo "   Size: $VM_SIZE (4 vCPUs, 16GB RAM)"
echo "   Image: $VM_IMAGE"
echo "   Location: $LOCATION"
echo "   This may take 3-5 minutes..."

az vm create \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --image $VM_IMAGE \
    --size $VM_SIZE \
    --admin-username $ADMIN_USERNAME \
    --ssh-key-values "${SSH_KEY_PATH}.pub" \
    --public-ip-sku Standard \
    --storage-sku StandardSSD_LRS

if [ $? -eq 0 ]; then
    echo "✅ Virtual machine created successfully"
else
    echo "❌ Failed to create virtual machine"
    echo ""
    echo "🔧 Troubleshooting options:"
    echo "1. Try different region: --location westus2"
    echo "2. Use smaller VM: Standard_B2ms (2 vCPUs, 8GB RAM)"
    echo "3. Check your quota: az vm list-usage --location $LOCATION"
    exit 1
fi

# Get VM public IP
echo "🌐 Getting VM public IP..."
VM_IP=$(az vm show -d -g $RESOURCE_GROUP -n $VM_NAME --query publicIps -o tsv)
if [ -n "$VM_IP" ]; then
    echo "✅ VM Public IP: $VM_IP"
else
    echo "⚠️  Getting IP address..."
    sleep 10
    VM_IP=$(az vm show -d -g $RESOURCE_GROUP -n $VM_NAME --query publicIps -o tsv)
    echo "✅ VM Public IP: $VM_IP"
fi

echo ""
echo "🎉 Student VM Deployment Complete!"
echo "=================================="
echo "VM Name: $VM_NAME"
echo "Public IP: $VM_IP"
echo "Username: $ADMIN_USERNAME"
echo "SSH Key: $SSH_KEY_PATH"
echo "VM Size: 4 vCPUs, 16GB RAM"
echo ""
echo "Connect to your VM:"
echo "ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
echo ""
echo "💡 Student Optimization Tips:"
echo "1. Use smaller vessel counts (4-6 vessels)"
echo "2. Set reasonable time limits (2-8 hours)"
echo "3. Stop VM when not in use to save credits"
echo "4. Monitor usage with: az consumption usage list"
echo ""
echo "Next steps:"
echo "1. Connect: ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
echo "2. Clone repo: git clone https://github.com/maercaestro/aegis-scheduler.git"
echo "3. Setup: cd aegis-scheduler && ./setup-vm.sh"
echo "4. Run: python main.py --vessel-count 4 --time-limit 14400"