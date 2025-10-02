"""
Solver management module for optimization problems.
Handles solver configuration, execution, and result handling.
"""

from pyomo.environ import SolverFactory, value
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from typing import Dict, Any, Optional, Tuple
import time
import sys
from contextlib import redirect_stdout
import os


class SolverManager:
    """Manages solver configuration and execution."""
    
    def __init__(self, solver_config: Optional[Dict[str, Any]] = None):
        """
        Initialize solver manager with configuration.
        
        Args:
            solver_config: Dictionary containing solver configuration
        """
        self.solver_config = solver_config or {
            "name": "highs",
            "options": {
                "threads": 4,
                "presolve": "on",
                "mip_rel_gap": 0.01
            }
        }
        self.solver = None
        self.results = None
        self.solve_time = 0
    
    def setup_solver(self, time_limit: int = 3600) -> bool:
        """
        Setup and configure the solver.
        
        Args:
            time_limit: Time limit for solver in seconds
            
        Returns:
            bool: True if solver setup successful, False otherwise
        """
        try:
            solver_name = self.solver_config.get("name", "highs")
            self.solver = SolverFactory(solver_name)
            
            if not self.solver.available():
                print(f"Warning: Solver {solver_name} is not available. Trying alternative solvers...")
                # Try alternative solvers
                alternatives = ["glpk", "cbc", "scip"]
                for alt_solver in alternatives:
                    try:
                        self.solver = SolverFactory(alt_solver)
                        if self.solver.available():
                            print(f"Using alternative solver: {alt_solver}")
                            solver_name = alt_solver
                            break
                    except Exception:
                        continue
                else:
                    print("Error: No suitable solver found")
                    return False
            
            print(f"Using solver: {solver_name}")
            
            # Set time limit
            self.solver.options["time_limit"] = time_limit
            
            # Apply solver-specific options
            for key, value in self.solver_config.get("options", {}).items():
                self.solver.options[key] = value
            
            print(f"Solver options: {dict(self.solver.options)}")
            return True
            
        except Exception as e:
            print(f"Error setting up solver: {e}")
            return False
    
    def solve_model(self, model, log_file_path: Optional[str] = None, 
                   tee: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Solve the optimization model.
        
        Args:
            model: Pyomo model to solve
            log_file_path: Path to save solver log (optional)
            tee: Whether to display solver output
            
        Returns:
            Tuple of (success, error_message)
        """
        if self.solver is None:
            return False, "Solver not initialized. Call setup_solver() first."
        
        try:
            start_time = time.time()
            
            if log_file_path:
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, "w") as f:
                    with redirect_stdout(f):
                        self.results = self.solver.solve(model, tee=tee)
            else:
                self.results = self.solver.solve(model, tee=tee)
            
            self.solve_time = time.time() - start_time
            
            # Check solver status
            return self._check_solution_status()
            
        except Exception as e:
            error_msg = f"Solver error: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def _check_solution_status(self) -> Tuple[bool, Optional[str]]:
        """
        Check the solution status and return success indicator.
        
        Returns:
            Tuple of (success, error_message)
        """
        if self.results is None:
            return False, "No results available"
        
        from pyomo.opt import SolverStatus, TerminationCondition
        
        solver_status = self.results.solver.status
        termination_condition = self.results.solver.termination_condition
        
        print(f"Solver Status: {solver_status}")
        print(f"Termination Condition: {termination_condition}")
        print(f"Solve Time: {self.solve_time:.2f} seconds")
        
        if solver_status == SolverStatus.ok:
            if termination_condition == TerminationCondition.optimal:
                print("Optimal solution found!")
                return True, None
            elif termination_condition == TerminationCondition.feasible:
                print("Feasible solution found (not necessarily optimal)")
                return True, None
            elif termination_condition == TerminationCondition.maxTimeLimit:
                print("Time limit reached - returning best solution found")
                return True, None
            else:
                error_msg = f"Solver terminated with condition: {termination_condition}"
                print(error_msg)
                return False, error_msg
        else:
            error_msg = f"Solver failed with status: {solver_status}"
            print(error_msg)
            return False, error_msg
    
    def get_objective_value(self, model) -> Optional[float]:
        """
        Get the objective value from the solved model.
        
        Args:
            model: Solved Pyomo model
            
        Returns:
            Objective value or None if not available
        """
        try:
            if hasattr(model, 'objective'):
                return value(model.objective)
            else:
                print("Warning: No objective function found in model")
                return None
        except Exception as e:
            print(f"Error getting objective value: {e}")
            return None
    
    def get_solve_statistics(self) -> Dict[str, Any]:
        """
        Get solver statistics and performance metrics.
        
        Returns:
            Dictionary containing solve statistics
        """
        stats = {
            "solve_time": self.solve_time,
            "solver_name": self.solver_config.get("name", "unknown"),
            "solver_options": self.solver_config.get("options", {}),
        }
        
        if self.results is not None:
            try:
                stats.update({
                    "solver_status": str(self.results.solver.status),
                    "termination_condition": str(self.results.solver.termination_condition),
                })
                
                # Add solver-specific statistics if available
                if hasattr(self.results.solver, 'time'):
                    stats["solver_time"] = self.results.solver.time
                
                if hasattr(self.results.problem, 'number_of_variables'):
                    stats["num_variables"] = self.results.problem.number_of_variables
                
                if hasattr(self.results.problem, 'number_of_constraints'):
                    stats["num_constraints"] = self.results.problem.number_of_constraints
                    
            except Exception as e:
                print(f"Warning: Could not extract all solver statistics: {e}")
        
        return stats
    
    def validate_solution(self, model) -> Dict[str, Any]:
        """
        Validate the solution and check constraint violations.
        
        Args:
            model: Solved Pyomo model
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "is_valid": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Check if solution exists
            if self.results is None:
                validation_results["is_valid"] = False
                validation_results["violations"].append("No solution available")
                return validation_results
            
            # Basic feasibility check
            from pyomo.opt import TerminationCondition
            if self.results.solver.termination_condition not in [
                TerminationCondition.optimal, 
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit
            ]:
                validation_results["is_valid"] = False
                validation_results["violations"].append(
                    f"Infeasible solution: {self.results.solver.termination_condition}"
                )
            
            # Check for unset variables (this might indicate solver issues)
            unset_vars = []
            for var in model.component_objects(ctype=pyo.Var):
                for index in var:
                    try:
                        val = value(var[index])
                        if val is None:
                            unset_vars.append(f"{var.name}[{index}]")
                    except ValueError:
                        unset_vars.append(f"{var.name}[{index}]")
            
            if unset_vars:
                validation_results["warnings"].append(
                    f"Found {len(unset_vars)} unset variables (first 5): {unset_vars[:5]}"
                )
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["violations"].append(f"Validation error: {str(e)}")
        
        return validation_results


def create_default_solver_config() -> Dict[str, Any]:
    """
    Create default solver configuration.
    
    Returns:
        Default solver configuration dictionary
    """
    return {
        "name": "highs", 
        "options": {
            "threads": 4,
            "presolve": "on",
            "mip_rel_gap": 0.01,
            "output_flag": True
        }
    }


def get_available_solvers() -> Dict[str, bool]:
    """
    Check which solvers are available on the system.
    
    Returns:
        Dictionary mapping solver names to availability status
    """
    common_solvers = ["highs", "glpk", "cbc", "scip", "gurobi", "cplex"]
    availability = {}
    
    for solver_name in common_solvers:
        try:
            solver = SolverFactory(solver_name)
            availability[solver_name] = solver.available()
        except Exception:
            availability[solver_name] = False
    
    return availability