# Aegis Scheduler - Vessel Routing & Crude Blending Optimization

A high-performance optimization system for vessel routing and crude oil blending operations, designed for deployment on Azure cloud infrastructure.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure account (for cloud deployment)
- Required data files in `test_data/` directory

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run system check:**
   ```bash
   python setup.py --check
   ```

3. **Run optimization:**
   ```bash
   python main.py --vessel-count 6 --optimization-type throughput --time-limit 3600
   ```

### Azure Deployment

1. **Create Azure deployment files:**
   ```bash
   python setup.py --create-azure-files
   ```

2. **Deploy to Azure:**
   ```bash
   ./deploy-azure.sh
   ```

## 📊 Features

- **High-Performance Optimization**: Multi-threaded solvers with advanced MILP capabilities
- **Modular Architecture**: Separate modules for data loading, modeling, solving, and result processing
- **Multiple Solvers**: Support for HiGHS, GLPK, CBC, SCIP, and commercial solvers
- **Cloud-Ready**: Optimized for Azure Container Instances with scalable compute resources
- **Comprehensive Results**: Detailed vessel routing schedules and crude blending plans

## 🏗️ Architecture

```
├── main.py                 # Main execution script
├── data_loader.py          # Data loading and preprocessing
├── optimization_model.py   # Pyomo model definition
├── solver_manager.py       # Solver configuration and execution
├── result_processor.py     # Result extraction and formatting
├── setup.py               # Configuration and deployment utilities
├── requirements.txt       # Python dependencies
└── test_data/            # Input data files
    ├── config.json
    ├── crude_availability.csv
    ├── crudes_info.csv
    ├── products_info.csv
    └── time_of_travel.csv
```

## 🎛️ Command Line Options

### Basic Usage
```bash
python main.py [OPTIONS]
```

### Key Options
- `--vessel-count N`: Number of vessels (default: 6)
- `--optimization-type TYPE`: 'margin' or 'throughput' (default: throughput)
- `--solver SOLVER`: Solver to use (default: highs)
- `--time-limit SECONDS`: Solver time limit (default: 3600)
- `--output-dir DIR`: Results directory (default: results/)
- `--data-path PATH`: Data directory (default: test_data/)

### Example Commands

**Optimize for maximum throughput with 8 vessels:**
```bash
python main.py --vessel-count 8 --optimization-type throughput --time-limit 7200
```

**Optimize for maximum margin with time limit:**
```bash
python main.py --optimization-type margin --time-limit 1800
```

**Use specific solver:**
```bash
python main.py --solver scip --threads 8
```

## 📈 Performance Optimization

### For Long-Running Optimizations:

1. **Increase time limit:**
   ```bash
   python main.py --time-limit 28800  # 8 hours
   ```

2. **Use multiple threads:**
   ```bash
   python main.py --threads 8
   ```

3. **Azure high-performance deployment:**
   - Deploy on Azure Container Instances with 8+ CPU cores
   - Use Premium storage for faster I/O
   - Consider Azure Batch for parallel scenario processing

## 🌥️ Azure Deployment Guide

### Method 1: Azure Container Instances (Recommended)

1. **Build and deploy:**
   ```bash
   # Create deployment files
   python setup.py --create-azure-files
   
   # Deploy to Azure
   ./deploy-azure.sh
   ```

2. **Monitor progress:**
   - Check Azure portal for container status
   - View logs in Container Instances console
   - Results are saved in container storage

### Method 2: Azure Virtual Machine

1. **Create VM with Docker:**
   ```bash
   # Create resource group
   az group create --name aegis-rg --location eastus
   
   # Create VM
   az vm create --resource-group aegis-rg --name aegis-vm --image Ubuntu2204 --size Standard_D8s_v3
   ```

2. **Setup and run:**
   ```bash
   # SSH to VM and install dependencies
   sudo apt update && sudo apt install -y python3-pip docker.io
   
   # Clone repository and run
   git clone <your-repo>
   cd aegis-scheduler
   pip3 install -r requirements.txt
   python3 main.py --vessel-count 6 --time-limit 86400  # 24 hours
   ```

## 📊 Output Files

The optimization generates several output files:

- `crude_blending_*.csv`: Detailed blending schedule by day/slot
- `vessel_routing_*.csv`: Vessel activities and routing schedule  
- `metrics_*.json`: Summary metrics (throughput, margin, demurrage)
- `solve_statistics.json`: Solver performance statistics
- `*.pkl`: Saved Pyomo model (if --save-model flag used)

## 🔧 Troubleshooting

### Common Issues

1. **No solver available:**
   ```bash
   python main.py --check-solvers
   python setup.py --install-deps
   ```

2. **Out of memory:**
   - Reduce vessel count or optimization horizon
   - Use Azure VMs with more RAM
   - Enable solver memory settings

3. **Slow convergence:**
   - Increase MIP gap tolerance: `--mip-gap 0.05`
   - Try different solver: `--solver scip`
   - Enable preprocessing: solver will do this automatically

### Performance Tips

- **For faster solutions**: Use `--mip-gap 0.05` for 5% optimality gap
- **For better solutions**: Increase `--time-limit` and use `--mip-gap 0.001`
- **For large problems**: Consider problem decomposition or scenario parallelization

## 📝 Data Format

### Required Files

1. **config.json**: Main configuration
2. **crude_availability.csv**: Crude availability by location and time window
3. **crudes_info.csv**: Crude types, margins, origins, opening inventory
4. **products_info.csv**: Product specifications and blending ratios
5. **time_of_travel.csv**: Travel times between locations

See `test_data/` directory for example formats.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section above
2. Run system check: `python setup.py --check`
3. Create GitHub issue with:
   - Command used
   - Error messages
   - System information
   - Data file samples (if applicable)

---

**Built for high-performance optimization on Azure Cloud** ☁️⚡

This is a standalone Python implementation of the refinery and vessel optimization system, converted from the original Jupyter notebook. The system optimizes crude oil vessel routing and refinery blending schedules to maximize either profit margins or throughput while minimizing demurrage costs.

## Features

- **Vessel Routing Optimization**: Determines optimal vessel schedules for crude oil pickup and delivery
- **Crude Blending Optimization**: Optimizes refinery production schedules and crude blend ratios
- **Dual Objectives**: Supports both margin maximization and throughput maximization
- **Constraint Management**: Handles complex operational constraints including vessel capacity, travel times, inventory limits, and discharge scheduling
- **EC2 Ready**: Modular design suitable for deployment on AWS EC2 instances

## Project Structure

```
aegis-scheduler/
├── main.py                    # Main optimization runner
├── config.py                  # Configuration parameters
├── example_runs.py            # Example optimization scenarios
├── data_loader.py             # Data loading and preprocessing
├── optimization_model.py      # Pyomo optimization model
├── solver_manager.py          # Solver configuration and execution
├── result_processor.py        # Results extraction and formatting
├── requirements.txt           # Python dependencies
├── test_data/                 # Input data directory
│   ├── config.json           # Optimization configuration
│   ├── crude_availability.csv # Crude availability by location/time
│   ├── crudes_info.csv       # Crude types and properties
│   ├── products_info.csv     # Product specifications
│   └── time_of_travel.csv    # Travel times between locations
└── results/                   # Output directory (created automatically)
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install optimization solver (choose one):**

   **Option A: HiGHS (Recommended)**
   ```bash
   pip install highspy
   ```

   **Option B: SCIP**
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get install scip
   pip install PySCIPOpt
   
   # On macOS with Homebrew:
   brew install scip
   pip install PySCIPOpt
   ```

   **Option C: GLPK**
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get install glpk-utils
   
   # On macOS with Homebrew:
   brew install glpk
   ```

## Usage

### Quick Start

Run the optimization with default parameters:

```bash
python main.py
```

This will:
- Load data from `test_data/` directory
- Run throughput optimization with 6 vessels
- Save results to `results/` directory
- Print summary metrics

### Configuration

Modify `config.py` to change optimization parameters:

```python
# Optimization parameters
VESSEL_COUNT = 6
OPTIMIZATION_TYPE = "throughput"  # "margin" or "throughput"
MAX_DEMURRAGE_LIMIT = 10
SCENARIO_NAME = "test_scenario"
```

### Example Runs

Use `example_runs.py` for different optimization scenarios:

```bash
python example_runs.py
```

### Custom Optimization

```python
from main import OptimizationRunner

# Create optimizer
runner = OptimizationRunner(data_path="test_data", output_path="results")

# Run optimization
results = runner.run_optimization(
    vessel_count=6,
    optimization_type="throughput",
    max_demurrage_limit=10,
    scenario_name="my_scenario"
)

# Access results
if results["status"] == "success":
    crude_blending_df = results["crude_blending_df"]
    vessel_routing_df = results["vessel_routing_df"]
    metrics = results["summary_metrics"]
```

## Input Data Format

### config.json
Contains optimization parameters:
- Inventory limits
- Vessel capacities
- Solver settings
- Plant capacity schedules

### crude_availability.csv
Crude availability by location and time window:
```csv
date_range,location,crude,volume,parcel_size
1-3 Oct,PM,Base,760000,400000
```

### crudes_info.csv
Crude properties and margins:
```csv
crudes,margin,origin,opening_inventory
Base,15.85,PM,350000
```

### products_info.csv
Product specifications and blend ratios:
```csv
product,max_per_day,crudes,ratios
Product1,50000,"['Base', 'A']","[0.7, 0.3]"
```

### time_of_travel.csv
Travel times between locations:
```csv
from,to,time_in_days
PM,Melaka,3
```

## Output Files

The optimization generates two main output files:

1. **Crude Blending Results** (`crude_blending_*.csv`)
   - Daily production schedules
   - Blend compositions
   - Inventory levels
   - Profit calculations

2. **Vessel Routing Results** (`vessel_routing_*.csv`)
   - Vessel activities by day
   - Loading/discharge schedules
   - Port visits and sailing times
   - Demurrage tracking

## Optimization Types

### Margin Optimization
- **Objective**: Maximize total profit (margin - demurrage costs)
- **Use case**: When profitability is the primary concern
- **Command**: Set `OPTIMIZATION_TYPE = "margin"`

### Throughput Optimization
- **Objective**: Maximize total production volume
- **Constraint**: Demurrage costs limited to specified threshold
- **Use case**: When production volume is prioritized
- **Command**: Set `OPTIMIZATION_TYPE = "throughput"`

## AWS EC2 Deployment

This system is designed to run on EC2 instances:

1. **Launch EC2 instance** with sufficient memory (recommend t3.large or larger)

2. **Install dependencies:**
   ```bash
   sudo yum update -y  # Amazon Linux
   sudo yum install -y python3 python3-pip git
   pip3 install -r requirements.txt
   ```

3. **Install solver:**
   ```bash
   # For HiGHS (recommended)
   pip3 install highspy
   ```

4. **Run optimization:**
   ```bash
   python3 main.py
   ```

5. **Automate with cron** (optional):
   ```bash
   # Run daily at 2 AM
   0 2 * * * cd /path/to/aegis-scheduler && python3 main.py
   ```

## Performance Notes

- **Memory**: Optimization models can use 2-8 GB RAM depending on scenario size
- **Time**: Typical solve times range from 1-30 minutes
- **Solvers**: HiGHS generally provides the best performance for this problem type
- **Scaling**: Model complexity increases with vessel count and planning horizon

## Troubleshooting

### Solver Issues
```bash
# Check if solver is available
python3 -c "from pyomo.environ import SolverFactory; print(SolverFactory('highs').available())"
```

### Memory Issues
- Reduce vessel count or planning horizon
- Use a larger EC2 instance type
- Increase solver time limit in config.json

### Infeasible Solutions
- Check data consistency in input files
- Verify vessel capacities vs. demand requirements
- Review travel time constraints

## License

This project is provided as-is for optimization and research purposes.

## Support

For issues or questions, please check:
1. Input data format and consistency
2. Solver installation and availability
3. Memory and computational requirements
4. EC2 instance specifications for deployment
