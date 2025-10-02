#!/bin/bash
# VM setup script - run this on the Azure VM after connecting

echo "🔧 Setting up Aegis Scheduler on Azure VM"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 and pip
echo "🐍 Installing Python 3.12..."
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Install system dependencies for solvers
echo "🔧 Installing solver dependencies..."
sudo apt install -y build-essential libglpk-dev glpk-utils coinor-cbc

# Install git if not present
sudo apt install -y git curl wget

# Create Python virtual environment
echo "🌐 Creating Python virtual environment..."
python3.12 -m venv aegis-env
source aegis-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages from requirements.txt
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Install additional solver packages
echo "🔍 Installing additional solvers..."
pip install PySCIPOpt

# Verify installation
echo "✅ Verifying installation..."
python --version
pip list | grep -E "(pandas|pyomo|numpy)"

# Check available solvers
echo "🔍 Checking available solvers..."
python main.py --check-solvers

# Create results directory
mkdir -p results
mkdir -p logs

# Set up environment activation script
cat > activate-aegis.sh << 'EOF'
#!/bin/bash
source aegis-env/bin/activate
echo "✅ Aegis environment activated"
echo "Ready to run optimization!"
echo ""
echo "Usage examples:"
echo "  python main.py --vessel-count 6 --time-limit 7200"
echo "  python main.py --optimization-type margin --time-limit 3600"
echo "  nohup python main.py --vessel-count 8 --time-limit 28800 > logs/optimization.log 2>&1 &"
EOF

chmod +x activate-aegis.sh

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "To activate the environment:"
echo "  source activate-aegis.sh"
echo ""
echo "To run optimization:"
echo "  python main.py --vessel-count 6 --time-limit 7200"
echo ""
echo "For long-running optimization (background):"
echo "  nohup python main.py --vessel-count 6 --time-limit 28800 > logs/optimization.log 2>&1 &"
echo ""
echo "To monitor background job:"
echo "  tail -f logs/optimization.log"