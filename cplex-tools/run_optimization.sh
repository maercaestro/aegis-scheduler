#!/bin/bash

# CPLEX Refinery Optimizer Runner Script
# This script activates the CPLEX environment and runs the optimization

echo "========================================"
echo "CPLEX Refinery Scheduler"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "cplex_refinery_optimizer.py" ]; then
    echo "Error: cplex_refinery_optimizer.py not found!"
    echo "Please run this script from the cplex-tools directory"
    exit 1
fi

# Check if CPLEX environment exists
if [ ! -d "../cplex" ]; then
    echo "Error: CPLEX environment not found!"
    echo "Please ensure the cplex virtual environment is set up"
    exit 1
fi

echo "Activating CPLEX environment..."
source ../cplex/bin/activate

echo "Python version: $(python --version)"
echo "CPLEX version: $(python -c 'import cplex; print(cplex.__version__)')"

echo "Starting optimization..."
echo "----------------------------------------"

# Run the optimization
python cplex_refinery_optimizer.py

echo "----------------------------------------"
echo "Optimization completed!"

# Check if results were generated
if [ -d "./results" ]; then
    echo "Results saved in: ./results/"
    echo "Files generated:"
    ls -la ./results/ | tail -n +2
else
    echo "No results directory found"
fi

echo "========================================"
