#!/bin/bash
# EC2 Setup Script for Refinery Optimization System
# This script sets up the complete environment on an Amazon Linux 2 or Ubuntu EC2 instance

set -e  # Exit on any error

echo "🚀 Setting up Refinery Optimization System on EC2"
echo "================================================="

# Update system packages
echo "📦 Updating system packages..."
if command -v yum &> /dev/null; then
    # Amazon Linux 2
    sudo yum update -y
    sudo yum install -y python3 python3-pip git gcc gcc-c++ make
    sudo yum install -y glpk glpk-devel  # GLPK solver
elif command -v apt &> /dev/null; then
    # Ubuntu
    sudo apt update
    sudo apt install -y python3 python3-pip git build-essential
    sudo apt install -y glpk-utils libglpk-dev  # GLPK solver
fi

# Upgrade pip
echo "🔧 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python requirements
echo "📚 Installing Python packages..."
python3 -m pip install -r requirements.txt

# Create results directory
echo "📁 Creating directories..."
mkdir -p results
mkdir -p logs

# Set permissions
chmod +x ec2_runner.py

# Test the installation
echo "🧪 Testing installation..."
python3 -c "import pyomo; import pandas; import numpy; import tqdm; print('✅ All packages installed successfully')"

# Test GLPK solver
echo "🔍 Testing GLPK solver..."
python3 -c "
from pyomo.environ import *
from pyomo.opt import SolverFactory
solver = SolverFactory('glpk')
if solver.available():
    print('✅ GLPK solver is available')
else:
    print('❌ GLPK solver not found')
    exit(1)
"

# Create a simple test run
echo "🏃 Running test optimization..."
python3 ec2_runner.py --vessels 3 --solutions 2 --output test_results --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 EC2 SETUP COMPLETED SUCCESSFULLY!"
    echo "✅ System is ready for production optimization"
    echo ""
    echo "🚀 To run optimization:"
    echo "   python3 ec2_runner.py --vessels 5 --solutions 5"
    echo ""
    echo "📋 Available options:"
    echo "   --config     Configuration file path (default: test_data/config.json)"
    echo "   --vessels    Number of vessels (default: 5)"
    echo "   --solutions  Number of vessel schedules to generate (default: 5)"
    echo "   --output     Output directory (default: results)"
    echo "   --verbose    Enable detailed logging"
else
    echo "❌ Setup test failed. Check the logs for details."
    exit 1
fi
