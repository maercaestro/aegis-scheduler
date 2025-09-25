# CPLEX Refinery Optimization Tools

This directory contains the CPLEX-based refinery scheduling optimization tools.

## Files

- `cplex_refinery_optimizer.py` - Main optimization script (MLflow removed, CPLEX-ready)
- `test_cplex_setup.py` - Test script to verify CPLEX installation and configuration
- `run_optimization.sh` - Bash script to easily run the optimization with proper environment

## Requirements

- Python 3.10 (required for CPLEX compatibility)
- CPLEX solver with Python API installed
- Virtual environment named `cplex` in parent directory
- Required Python packages: pyomo, pandas, cplex

## Setup

1. Ensure CPLEX environment is activated:
   ```bash
   source ../cplex/bin/activate
   ```

2. Verify installation:
   ```bash
   python test_cplex_setup.py
   ```

## Running the Optimization

### Method 1: Using the run script (recommended)
```bash
./run_optimization.sh
```

### Method 2: Manual execution
```bash
source ../cplex/bin/activate
python cplex_refinery_optimizer.py
```

## Configuration

Edit the parameters at the bottom of `cplex_refinery_optimizer.py`:

- `VESSEL_COUNT`: Number of vessels available for scheduling
- `OPTIMIZATION_GOAL`: Either "margin" or "throughput"
- `MAX_DEMURRAGE_DAYS_LIMIT`: Maximum demurrage days (for throughput optimization)

## Key Changes from Original

1. **Removed MLflow dependencies** - All MLflow logging and experiment tracking removed
2. **CPLEX Direct Interface** - Uses `cplex_direct` for better integration with Python API
3. **Improved error handling** - Better handling of infeasible solutions
4. **Local file paths** - Uses local `./results/` directory instead of lakehouse paths
5. **Timestamped outputs** - Results files include timestamps to avoid overwrites
6. **Enhanced solver options** - Optimized CPLEX settings for better performance

## Output

Results are saved in the `./results/` directory with timestamped filenames:
- `crude_blending_[config]_[timestamp].csv` - Production schedule
- `vessel_routing_[config]_[timestamp].csv` - Vessel activities
- `optimization_log_[config]_[timestamp].txt` - Solver output log
- `model_[config]_[timestamp].pkl` - Pickled Pyomo model

## Troubleshooting

1. **CPLEX not found**: Ensure CPLEX is properly installed and licensed
2. **Import errors**: Verify you're using Python 3.10 and the correct virtual environment
3. **Solver issues**: Run the test script to diagnose CPLEX problems
4. **Data loading errors**: Ensure test data files are available in `../test_data/`
