#!/bin/bash

# Quick connect to Azure VM and run setup
VM_IP="172.173.202.235"
SSH_KEY="/Users/abuhuzaifahbidin/.ssh/aegis_minimal_key"
VM_USER="azureuser"

echo "🔗 Connecting to Azure VM..."
echo "VM: $VM_USER@$VM_IP"
echo ""

# Connect and provide setup instructions
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" << 'EOF'
echo "🎯 Connected to Azure VM!"
echo "========================"
echo ""
echo "📋 Setup Instructions:"
echo "1. Run the git-based setup:"
echo "   curl -sSL https://raw.githubusercontent.com/maercaestro/aegis-scheduler/main/vm-setup-git.sh | bash"
echo ""
echo "2. Or manual setup:"
echo "   git clone https://github.com/maercaestro/aegis-scheduler.git"
echo "   cd aegis-scheduler"
echo "   chmod +x vm-setup-git.sh"
echo "   ./vm-setup-git.sh"
echo ""
echo "💡 This will set up both manual and systemd process management!"
echo ""
EOF