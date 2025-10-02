#!/bin/bash
# Local environment activation script for Aegis Scheduler

echo "🚀 Activating Aegis Scheduler Environment"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the aegis-scheduler directory."
    exit 1
fi

# Activate aegis virtual environment
if [ -d "aegis" ]; then
    echo "🐍 Activating Python virtual environment 'aegis'..."
    source aegis/bin/activate
    
    if [[ "$VIRTUAL_ENV" == *"aegis"* ]]; then
        echo "✅ Aegis environment activated"
    else
        echo "❌ Failed to activate aegis environment"
        echo "   Please check if aegis/bin/activate exists"
        exit 1
    fi
else
    echo "❌ Aegis virtual environment not found."
    echo "   Please make sure you're in the aegis-scheduler directory"
    echo "   and the 'aegis' folder exists with bin/activate"
    exit 1
fi

# Verify Python version and packages
echo "🔍 Verifying environment..."
python --version
echo "HiGHS solver available: $(python -c 'from pyomo.environ import SolverFactory; print(SolverFactory("highs").available())' 2>/dev/null)"

echo ""
echo "🎯 Ready to run optimization!"
echo "============================="
echo ""
echo "Quick test (5 minutes):"
echo "  python main.py --vessel-count 6 --time-limit 300"
echo ""
echo "Production run (2 hours):"
echo "  python main.py --vessel-count 6 --time-limit 7200"
echo ""
echo "Long optimization (8 hours):"
echo "  python main.py --vessel-count 8 --time-limit 28800"
echo ""
echo "Check available solvers:"
echo "  python main.py --check-solvers"