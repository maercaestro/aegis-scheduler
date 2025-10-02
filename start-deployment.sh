#!/bin/bash
# Aegis Environment Deployment Starter

echo "🚀 Starting Azure Deployment from Aegis Environment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Please run this from the aegis-scheduler directory"
    echo "cd /Users/abuhuzaifahbidin/Documents/GitHub/aegis-scheduler"
    exit 1
fi

# Activate aegis environment
echo "🐍 Activating Aegis environment..."
source aegis/bin/activate

if [[ "$CONDA_DEFAULT_ENV" != "aegis" ]]; then
    echo "❌ Failed to activate aegis environment"
    echo "Please manually run: conda activate aegis"
    exit 1
fi

echo "✅ Aegis environment activated"

# Quick verification that optimization works
echo "🔍 Quick system check..."
python -c "
import pandas as pd
import pyomo.environ as pyo
from solver_manager import get_available_solvers

print('✅ Pandas version:', pd.__version__)
print('✅ Pyomo available')

solvers = get_available_solvers()
available = [s for s, a in solvers.items() if a]
if available:
    print('✅ Available solvers:', ', '.join(available))
else:
    print('⚠️  No solvers available - will install SCIP on VM')
"

if [ $? -eq 0 ]; then
    echo "✅ Local environment check passed"
else
    echo "❌ Local environment check failed"
    exit 1
fi

echo ""
echo "🚀 Ready for Azure deployment!"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Install Azure CLI (if needed): brew install azure-cli"
echo "2. Deploy minimal VM: ./deploy-minimal-vm.sh"
echo "3. Or run automated setup: ./install-and-deploy.sh"
echo ""

# Ask user what they want to do
echo "What would you like to do?"
echo "1. Install Azure CLI and deploy automatically"
echo "2. Deploy minimal VM (assumes Azure CLI installed)"
echo "3. Deploy standard student VM (assumes Azure CLI installed)"
echo "4. Just check if Azure CLI is installed"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Running automated setup..."
        ./install-and-deploy.sh
        ;;
    2)
        if command -v az &> /dev/null; then
            ./deploy-minimal-vm.sh
        else
            echo "❌ Azure CLI not found. Please choose option 1 first."
        fi
        ;;
    3)
        if command -v az &> /dev/null; then
            ./deploy-student-vm.sh
        else
            echo "❌ Azure CLI not found. Please choose option 1 first."
        fi
        ;;
    4)
        if command -v az &> /dev/null; then
            echo "✅ Azure CLI is installed"
            az --version
        else
            echo "❌ Azure CLI not found"
            echo "Install with: brew install azure-cli"
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac