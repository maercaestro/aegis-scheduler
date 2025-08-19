# Configuration file for optimization runs
# This file allows you to easily modify optimization parameters

# Optimization parameters
VESSEL_COUNT = 6
OPTIMIZATION_TYPE = "throughput"  # Options: "margin", "throughput"
MAX_DEMURRAGE_LIMIT = 10  # Only used for throughput optimization
SCENARIO_NAME = "test_scenario"

# Paths
DATA_PATH = "test_data"
OUTPUT_PATH = "results"

# Solver preferences (optional - will use config.json if not specified)
SOLVER_NAME = "highs"  # Options: "highs", "scip", "glpk"
SOLVER_TIME_LIMIT = 3600  # seconds
MIP_GAP = 0.01  # 1% optimality gap
