#!/bin/bash
# Azure CLI installation and deployment guide for Aegis environment

echo "🚀 Aegis Scheduler - Azure Setup & Deployment"
echo "=============================================="

# First, let's make sure we're in the right environment
echo "🔍 Checking current environment..."
if [ -d "aegis" ]; then
    echo "⚠️  Activating aegis virtual environment..."
    source aegis/bin/activate
    if [[ "$VIRTUAL_ENV" == *"aegis"* ]]; then
        echo "✅ Aegis environment active"
    else
        echo "❌ Failed to activate aegis environment"
        echo "Please run: source aegis/bin/activate"
        exit 1
    fi
else
    echo "❌ Aegis environment not found"
    echo "Please make sure you're in the aegis-scheduler directory and aegis/ folder exists"
    exit 1
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo ""
    echo "📥 Installing Azure CLI..."
    
    # Install Azure CLI on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "Using Homebrew to install Azure CLI..."
            brew install azure-cli
        else
            echo "Installing Azure CLI using curl..."
            curl -L https://aka.ms/InstallAzureCli | bash
        fi
    else
        echo "Installing Azure CLI using curl..."
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    fi
    
    # Verify installation
    if command -v az &> /dev/null; then
        echo "✅ Azure CLI installed successfully"
        az --version
    else
        echo "❌ Azure CLI installation failed"
        echo "Please install manually: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
else
    echo "✅ Azure CLI already installed"
    az --version
fi

echo ""
echo "🔐 Logging into Azure..."
az login

echo ""
echo "📋 Available subscriptions:"
az account list --output table

echo ""
echo "🎯 Ready for deployment!"
echo "======================="
echo ""
echo "Choose your deployment option:"
echo "1. Minimal VM (2 vCPUs, 4GB RAM) - Fits any student quota"
echo "2. Standard VM (4 vCPUs, 16GB RAM) - Better performance"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "Deploying minimal VM..."
        ./deploy-minimal-vm.sh
        ;;
    2)
        echo "Deploying standard student VM..."
        ./deploy-student-vm.sh
        ;;
    *)
        echo "Invalid choice. Please run script again."
        exit 1
        ;;
esac