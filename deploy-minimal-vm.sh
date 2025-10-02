#!/bin/bash
# Ultra-lightweight VM for Azure Student subscription - Minimum resources

echo "💡 Aegis Scheduler - Minimal Student VM Deployment"
echo "================================================="

# Minimal configuration for student accounts
RESOURCE_GROUP="aegis-minimal-rg"
LOCATION="eastus"
VM_NAME="aegis-minimal-vm"
VM_SIZE="Standard_B2s"  # 2 vCPUs, 4GB RAM - very student budget friendly
VM_IMAGE="Ubuntu2204"
ADMIN_USERNAME="azureuser"

# Alternative locations if eastus is full
ALT_LOCATIONS=("westus2" "centralus" "westeurope")

echo "🎓 Student-friendly minimal deployment"
echo "   VM Size: Standard_B2s (2 vCPUs, 4GB RAM)"
echo "   Storage: Standard HDD (cheapest option)"
echo "   Optimized for Azure Student credits"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Error: Azure CLI not found. Please install Azure CLI first."
    exit 1
fi

# Login check
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
fi

# Create resource group
echo "📁 Creating resource group: $RESOURCE_GROUP..."
az group create --name $RESOURCE_GROUP --location $LOCATION
if [ $? -ne 0 ]; then
    echo "⚠️  Failed to create resource group in $LOCATION, trying alternative locations..."
    for alt_loc in "${ALT_LOCATIONS[@]}"; do
        echo "Trying $alt_loc..."
        if az group create --name $RESOURCE_GROUP --location $alt_loc; then
            LOCATION=$alt_loc
            echo "✅ Resource group created in $LOCATION"
            break
        fi
    done
fi

# Generate SSH key
SSH_KEY_PATH="$HOME/.ssh/aegis_minimal_key"
if [ ! -f "$SSH_KEY_PATH" ]; then
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "aegis-minimal-vm"
fi

# Create minimal VM
echo "💻 Creating minimal VM (this fits any student quota)..."
az vm create \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --image $VM_IMAGE \
    --size $VM_SIZE \
    --admin-username $ADMIN_USERNAME \
    --ssh-key-values "${SSH_KEY_PATH}.pub" \
    --public-ip-sku Basic \
    --storage-sku Standard_LRS

if [ $? -eq 0 ]; then
    VM_IP=$(az vm show -d -g $RESOURCE_GROUP -n $VM_NAME --query publicIps -o tsv)
    echo ""
    echo "✅ Minimal VM deployed successfully!"
    echo "================================="
    echo "VM IP: $VM_IP"
    echo "SSH Key: $SSH_KEY_PATH"
    echo ""
    echo "Connect:"
    echo "ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
    echo ""
    echo "💡 For minimal VM, use these optimization settings:"
    echo "  --vessel-count 3"
    echo "  --time-limit 7200"
    echo "  --threads 2"
else
    echo "❌ Deployment failed"
fi