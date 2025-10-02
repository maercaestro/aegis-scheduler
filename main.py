#!/usr/bin/env python3
"""
Main execution script for vessel routing and crude blending optimization.
Orchestrates the complete optimization workflow from data loading to result processing.
"""

import argparse
import sys
import os
import time
import json
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_data
from optimization_model import OptimizationModel
from solver_manager import SolverManager, get_available_solvers
from result_processor import ResultProcessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vessel routing and crude blending optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and configuration
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='test_data/',
        help='Path to data directory containing config.json and CSV files'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results/',
        help='Directory to save optimization results'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--vessel-count', 
        type=int, 
        default=6,
        help='Number of vessels to optimize'
    )
    
    parser.add_argument(
        '--optimization-type', 
        type=str, 
        choices=['margin', 'throughput'],
        default='throughput',
        help='Type of optimization objective'
    )
    
    parser.add_argument(
        '--max-demurrage-limit', 
        type=int, 
        default=10,
        help='Maximum demurrage limit for throughput optimization'
    )
    
    # Solver configuration
    parser.add_argument(
        '--solver', 
        type=str, 
        default='highs',
        help='Solver to use (highs, glpk, cbc, scip, etc.)'
    )
    
    parser.add_argument(
        '--time-limit', 
        type=int, 
        default=3600,
        help='Solver time limit in seconds'
    )
    
    parser.add_argument(
        '--mip-gap', 
        type=float, 
        default=0.01,
        help='MIP relative gap tolerance'
    )
    
    parser.add_argument(
        '--threads', 
        type=int, 
        default=4,
        help='Number of solver threads'
    )
    
    # Output options
    parser.add_argument(
        '--save-model', 
        action='store_true',
        help='Save the Pyomo model to pickle file'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        help='Path to save solver log file'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress solver output'
    )
    
    parser.add_argument(
        '--check-solvers', 
        action='store_true',
        help='Check available solvers and exit'
    )
    
    return parser.parse_args()


def check_available_solvers():
    """Check and display available solvers."""
    print("Checking available solvers...")
    solvers = get_available_solvers()
    
    print("\nSolver availability:")
    print("-" * 30)
    for solver, available in solvers.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"{solver:10s} - {status}")
    
    available_solvers = [s for s, avail in solvers.items() if avail]
    if available_solvers:
        print(f"\nRecommended solver: {available_solvers[0]}")
    else:
        print("\nWarning: No common solvers found. You may need to install a solver.")


def setup_logging(quiet: bool, log_file: Optional[str] = None):
    """Setup logging configuration."""
    if quiet:
        # Redirect stdout to suppress solver output
        import logging
        logging.basicConfig(level=logging.WARNING)
    
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Check solvers if requested
    if args.check_solvers:
        check_available_solvers()
        return 0
    
    print("="*70)
    print("VESSEL ROUTING & CRUDE BLENDING OPTIMIZATION")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Vessel count: {args.vessel_count}")
    print(f"Optimization type: {args.optimization_type}")
    print(f"Solver: {args.solver}")
    print(f"Time limit: {args.time_limit}s")
    print("="*70)
    
    try:
        # Setup logging
        setup_logging(args.quiet, args.log_file)
        
        # Step 1: Load data
        print("\n1. Loading data...")
        start_time = time.time()
        
        if not os.path.exists(args.data_path):
            print(f"Error: Data path '{args.data_path}' does not exist")
            return 1
        
        data = load_all_data(args.data_path)
        load_time = time.time() - start_time
        print(f"   Data loaded successfully ({load_time:.2f}s)")
        print(f"   Found {len(data['crudes'])} crude types")
        print(f"   Found {len(data['locations'])} locations")
        print(f"   Optimization period: {data['config']['DAYS']['start']} to {data['config']['DAYS']['end']} days")
        
        # Step 2: Create optimization model
        print("\n2. Building optimization model...")
        start_time = time.time()
        
        optimizer = OptimizationModel(
            data=data,
            vessel_count=args.vessel_count,
            optimization_type=args.optimization_type,
            max_demurrage_limit=args.max_demurrage_limit
        )
        
        model = optimizer.create_model()
        build_time = time.time() - start_time
        print(f"   Model built successfully ({build_time:.2f}s)")
        
        # Print model statistics
        num_vars = sum(1 for _ in model.component_objects(ctype=pyo.Var))
        num_constraints = sum(1 for _ in model.component_objects(ctype=pyo.Constraint))
        print(f"   Variables: {num_vars}")
        print(f"   Constraints: {num_constraints}")
        
        # Step 3: Setup and run solver
        print("\n3. Setting up solver...")
        solver_config = {
            "name": args.solver,
            "options": {
                "threads": args.threads,
                "presolve": "on",
                "mip_rel_gap": args.mip_gap,
                "output_flag": not args.quiet
            }
        }
        
        solver_manager = SolverManager(solver_config)
        
        if not solver_manager.setup_solver(args.time_limit):
            print("Error: Failed to setup solver")
            return 1
        
        print("\n4. Solving optimization problem...")
        solve_start = time.time()
        
        success, error_msg = solver_manager.solve_model(
            model, 
            log_file_path=args.log_file,
            tee=not args.quiet
        )
        
        solve_time = time.time() - solve_start
        
        if not success:
            print(f"Error: Optimization failed - {error_msg}")
            return 1
        
        print(f"   Optimization completed successfully ({solve_time:.2f}s)")
        
        # Get objective value
        objective_value = solver_manager.get_objective_value(model)
        if objective_value is not None:
            print(f"   Objective value: {objective_value:,.2f}")
        
        # Step 4: Process and save results
        print("\n5. Processing results...")
        result_processor = ResultProcessor(model, data, data['config'])
        
        # Save results
        max_transitions = data['config'].get('MaxTransitions')
        file_paths = result_processor.save_results(
            output_dir=args.output_dir,
            vessel_count=args.vessel_count,
            optimization_type=args.optimization_type,
            max_demurrage_limit=args.max_demurrage_limit if args.optimization_type == 'throughput' else None,
            max_transitions=max_transitions
        )
        
        print("   Results saved:")
        for result_type, path in file_paths.items():
            print(f"     {result_type}: {path}")
        
        # Print summary
        result_processor.print_summary()
        
        # Save solver statistics
        stats = solver_manager.get_solve_statistics()
        stats_file = os.path.join(args.output_dir, 'solve_statistics.json')
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"   Solver statistics saved: {stats_file}")
        except Exception as e:
            print(f"   Warning: Could not save solver statistics: {e}")
        
        # Validate solution
        print("\n6. Validating solution...")
        validation = solver_manager.validate_solution(model)
        
        if validation['is_valid']:
            print("   ✓ Solution validation passed")
        else:
            print("   ✗ Solution validation failed:")
            for violation in validation['violations']:
                print(f"     - {violation}")
        
        if validation['warnings']:
            print("   Warnings:")
            for warning in validation['warnings']:
                print(f"     - {warning}")
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Import pyomo here to avoid issues with command line parsing
    import pyomo.environ as pyo
    
    exit_code = main()
    sys.exit(exit_code)