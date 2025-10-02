#!/bin/bash

# Transfer optimization files to Azure VM
# Usage: ./transfer-to-vm.sh

set -e

# VM connection details
VM_IP="172.173.202.235"
SSH_KEY="/Users/abuhuzaifahbidin/.ssh/aegis_minimal_key"
VM_USER="azureuser"

echo "📁 Transferring optimization files to Azure VM..."
echo "VM: $VM_USER@$VM_IP"
echo "=================================="

# Files to transfer
FILES=(
    "data_loader.py"
    "optimization_model.py" 
    "solver_manager.py"
    "result_processor.py"
    "main.py"
    "requirements.txt"
    "setup-vm.sh"
    "test_data/"
)

# Create project directory on VM
echo "📂 Creating project directory on VM..."
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "mkdir -p ~/aegis-scheduler"

# Transfer each file/directory
for item in "${FILES[@]}"; do
    if [ -e "$item" ]; then
        echo "📤 Transferring: $item"
        if [ -d "$item" ]; then
            # Transfer directory
            scp -i "$SSH_KEY" -r "$item" "$VM_USER@$VM_IP:~/aegis-scheduler/"
        else
            # Transfer file
            scp -i "$SSH_KEY" "$item" "$VM_USER@$VM_IP:~/aegis-scheduler/"
        fi
    else
        echo "⚠️  File not found: $item"
    fi
done

echo ""
echo "✅ File transfer complete!"
echo ""
echo "🔗 Next steps:"
echo "1. Connect to VM:"
echo "   ssh -i $SSH_KEY $VM_USER@$VM_IP"
echo ""
echo "2. Setup environment:"
echo "   cd ~/aegis-scheduler"
echo "   chmod +x setup-vm.sh"
echo "   ./setup-vm.sh"
echo ""
echo "3. Run optimization (minimal VM settings):"
echo "   source aegis-env/bin/activate"
echo "   python main.py --vessel-count 3 --time-limit 7200 --threads 2"