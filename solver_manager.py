"""
Solver management utilities for the optimization system.
Handles solver configuration and execution.
"""

from pyomo.environ import SolverFactory
from contextlib import redirect_stdout
import sys
import os


class SolverManager:
    """Manages solver configuration and execution."""
    
    def __init__(self, config: dict):
        """
        Initialize solver manager.
        
        Args:
            config: Configuration dictionary containing solver settings
        """
        self.config = config
        
    def get_solver_config(self) -> dict:
        """Get solver configuration. Defaults to HiGHS if available."""
        # Default solver configuration
        default_config = {
            "name": "highs",
            "options": {
                "threads": 4,
                "presolve": "on",
                "mip_rel_gap": 0.01
            }
        }
        
        # Check if solver config exists in config
        if "solver" in self.config:
            for solver_cfg in self.config["solver"]:
                if solver_cfg.get("use", False):
                    return {
                        "name": solver_cfg["name"],
                        "options": solver_cfg.get("options", {})
                    }
        
        return default_config
        
    def get_enabled_solver(self, solver_cfg: dict = None):
        """Get the configured solver with all options applied."""
        if solver_cfg is None:
            solver_cfg = self.get_solver_config()
            
        solver_name = solver_cfg.get("name")
        solver = SolverFactory(solver_name)

        print(f"Using solver: {solver_name}")
 
        # Set solver-specific options
        if solver_name.lower() == "highs":
            # HiGHS specific options
            solver.options["time_limit"] = self.config.get("solver_time_limit_seconds", 3600)
            # Set other HiGHS options
            for key, value in solver_cfg.get("options", {}).items():
                solver.options[key] = value
        elif solver_name.lower() == "glpk":
            # GLPK doesn't support time_limit option the same way
            # Use tmlim for time limit in seconds
            solver.options["tmlim"] = self.config.get("solver_time_limit_seconds", 3600)
        else:
            # Generic solver options
            solver.options["time_limit"] = self.config.get("solver_time_limit_seconds", 3600)
            for key, value in solver_cfg.get("options", {}).items():
                solver.options[key] = value

        print(f"Solver options: {solver.options}")
        return solver
        
    def solve_model(self, model, solver_cfg: dict = None, log_file_path: str = None, 
                   scenario_name: str = "test", vessel_count: int = 6, 
                   optimization_type: str = "throughput", max_demurrage_limit: int = 10):
        """
        Solve the optimization model.
        
        Args:
            model: Pyomo model to solve
            solver_cfg: Solver configuration
            log_file_path: Path to save solver log
            scenario_name: Name of the scenario
            vessel_count: Number of vessels
            optimization_type: Type of optimization
            max_demurrage_limit: Maximum demurrage limit
            
        Returns:
            Solver results object
        """
        solver = self.get_enabled_solver(solver_cfg)
        
        # Generate log file path if not provided
        if log_file_path is None:
            os.makedirs("results", exist_ok=True)
            if optimization_type == 'throughput':
                log_file_path = f'results/{optimization_type}_optimization_log_{vessel_count}_vessels_{self.config["DAYS"]["end"]}_days_{self.config["MaxTransitions"]}_transitions_{max_demurrage_limit}_demurrages.txt'
            else:
                log_file_path = f'results/{optimization_type}_optimization_log_{vessel_count}_vessels_{self.config["DAYS"]["end"]}_days_{self.config["MaxTransitions"]}_transitions.txt'

        print(f"Starting optimization...")
        print(f"Log file: {log_file_path}")
        
        try:
            with open(log_file_path, "w") as f:
                with redirect_stdout(f):
                    results = solver.solve(model, tee=True)
            
            print("Optimization completed successfully!")
            return results
            
        except Exception as e:
            error_msg = f"Error Occurred: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
