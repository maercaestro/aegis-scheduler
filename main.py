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
import logging
from typing import Optional, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_data
from optimization_model import OptimizationModel
from solver_manager import SolverManager, get_available_solvers
from result_processor import ResultProcessor


def load_config_file(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading configuration file {config_path}: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vessel routing and crude blending optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file option
    parser.add_argument(
        '--config-file', 
        type=str,
        help='JSON configuration file path (overrides command line arguments)'
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
    
    # Scenario selection
    parser.add_argument(
        '--scenario', 
        type=int, 
        default=1,
        help='Scenario number to optimize'
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
    
    # Service mode options
    parser.add_argument(
        '--service-mode', 
        action='store_true',
        help='Run in service mode (enhanced logging, error handling)'
    )
    
    parser.add_argument(
        '--service-name', 
        type=str,
        help='Service name for logging identification'
    )
    
    return parser.parse_args()


def apply_config_overrides(args, config: Dict[str, Any]):
    """Apply configuration file values to arguments."""
    # Map config file keys to argument attributes
    config_mapping = {
        'scenario': 'scenario',
        'vessel_count': 'vessel_count', 
        'optimization_type': 'optimization_type',
        'max_demurrage_limit': 'max_demurrage_limit',
        'solver': 'solver',
        'time_limit': 'time_limit',
        'mip_gap': 'mip_gap',
        'threads': 'threads',
        'data_path': 'data_path',
        'output_dir': 'output_dir',
        'save_model': 'save_model',
        'log_file': 'log_file',
        'quiet': 'quiet',
        'service_name': 'service_name'
    }
    
    for config_key, arg_attr in config_mapping.items():
        if config_key in config and hasattr(args, arg_attr):
            setattr(args, arg_attr, config[config_key])
            logging.info(f"Config override: {arg_attr} = {config[config_key]}")
    
    return args


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


def setup_logging(quiet: bool, service_mode: bool = False, service_name: Optional[str] = None, log_file: Optional[str] = None):
    """Setup logging configuration."""
    # Determine log level
    if quiet:
        log_level = logging.WARNING
    elif service_mode:
        log_level = logging.INFO
    else:
        log_level = logging.INFO
    
    # Setup logging format
    if service_mode:
        log_format = f'%(asctime)s - {service_name or "aegis"} - %(levelname)s - %(message)s'
    else:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(file_handler)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration file if provided
    if args.config_file:
        config = load_config_file(args.config_file)
        if config is None:
            print(f"Error: Failed to load configuration file: {args.config_file}")
            return 1
        
        # Apply config overrides
        args = apply_config_overrides(args, config)
    
    # Setup logging
    setup_logging(args.quiet, args.service_mode, args.service_name, args.log_file)
    
    # Check solvers if requested
    if args.check_solvers:
        check_available_solvers()
        return 0
    
    # Service mode vs interactive mode output
    if args.service_mode:
        logging.info("Starting Aegis Scheduler optimization service")
        logging.info(f"Service name: {args.service_name or 'aegis-optimization'}")
        logging.info(f"Configuration: vessels={args.vessel_count}, type={args.optimization_type}, solver={args.solver}")
        logging.info(f"Time limit: {args.time_limit}s, Threads: {args.threads}")
    else:
        print("="*70)
        print("VESSEL ROUTING & CRUDE BLENDING OPTIMIZATION")
        print("="*70)
        print(f"Data path: {args.data_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Scenario: {args.scenario}")
        print(f"Vessel count: {args.vessel_count}")
        print(f"Optimization type: {args.optimization_type}")
        print(f"Solver: {args.solver}")
        print(f"Time limit: {args.time_limit}s")
        print("="*70)
    
    try:
        
        # Step 1: Load data
        log_msg = "Loading data..."
        if args.service_mode:
            logging.info(log_msg)
        else:
            print(f"\n1. {log_msg}")
        
        start_time = time.time()
        
        if not os.path.exists(args.data_path):
            error_msg = f"Data path '{args.data_path}' does not exist"
            if args.service_mode:
                logging.error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        data = load_all_data(args.data_path)
        load_time = time.time() - start_time
        
        success_msg = f"Data loaded successfully ({load_time:.2f}s)"
        details = [
            f"Found {len(data['crudes'])} crude types",
            f"Found {len(data['locations'])} locations", 
            f"Optimization period: {data['config']['DAYS']['start']} to {data['config']['DAYS']['end']} days"
        ]
        
        if args.service_mode:
            logging.info(success_msg)
            for detail in details:
                logging.info(f"  {detail}")
        else:
            print(f"   {success_msg}")
            for detail in details:
                print(f"   {detail}")
        
        # Step 2: Create optimization model
        log_msg = "Building optimization model..."
        if args.service_mode:
            logging.info(log_msg)
        else:
            print(f"\n2. {log_msg}")
        
        start_time = time.time()
        
        optimizer = OptimizationModel(
            data=data,
            vessel_count=args.vessel_count,
            optimization_type=args.optimization_type,
            max_demurrage_limit=args.max_demurrage_limit
        )
        
        model = optimizer.create_model()
        build_time = time.time() - start_time
        
        success_msg = f"Model built successfully ({build_time:.2f}s)"
        
        # Get model statistics
        num_vars = sum(1 for _ in model.component_objects(ctype=pyo.Var))
        num_constraints = sum(1 for _ in model.component_objects(ctype=pyo.Constraint))
        
        if args.service_mode:
            logging.info(success_msg)
            logging.info(f"  Variables: {num_vars}, Constraints: {num_constraints}")
        else:
            print(f"   {success_msg}")
            print(f"   Variables: {num_vars}")
            print(f"   Constraints: {num_constraints}")
        
        # Step 3: Setup and run solver
        log_msg = "Setting up solver..."
        if args.service_mode:
            logging.info(log_msg)
        else:
            print(f"\n3. {log_msg}")
        
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
            error_msg = "Failed to setup solver"
            if args.service_mode:
                logging.error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        log_msg = "Solving optimization problem..."
        if args.service_mode:
            logging.info(log_msg)
        else:
            print(f"\n4. {log_msg}")
        
        solve_start = time.time()
        
        success, error_msg = solver_manager.solve_model(
            model, 
            log_file_path=args.log_file,
            tee=not args.quiet
        )
        
        solve_time = time.time() - solve_start
        
        if not success:
            error_msg = f"Optimization failed - {error_msg}"
            if args.service_mode:
                logging.error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        success_msg = f"Optimization completed successfully ({solve_time:.2f}s)"
        if args.service_mode:
            logging.info(success_msg)
        else:
            print(f"   {success_msg}")
        
        # Get objective value
        objective_value = solver_manager.get_objective_value(model)
        if objective_value is not None:
            obj_msg = f"Objective value: {objective_value:,.2f}"
            if args.service_mode:
                logging.info(obj_msg)
            else:
                print(f"   {obj_msg}")
        
        # Step 4: Process and save results
        log_msg = "Processing results..."
        if args.service_mode:
            logging.info(log_msg)
        else:
            print(f"\n5. {log_msg}")
        
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
        
        if args.service_mode:
            logging.info("Results saved:")
            for result_type, path in file_paths.items():
                logging.info(f"  {result_type}: {path}")
        else:
            print("   Results saved:")
            for result_type, path in file_paths.items():
                print(f"     {result_type}: {path}")
        
        # Print/log summary
        if args.service_mode:
            # For service mode, capture summary as log messages
            logging.info("Optimization summary completed")
        else:
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
        log_msg = "Validating solution..."
        if args.service_mode:
            logging.info(log_msg)
        else:
            print(f"\n6. {log_msg}")
        
        validation = solver_manager.validate_solution(model)
        
        if validation['is_valid']:
            success_msg = "Solution validation passed"
            if args.service_mode:
                logging.info(f"✓ {success_msg}")
            else:
                print(f"   ✓ {success_msg}")
        else:
            error_msg = "Solution validation failed"
            if args.service_mode:
                logging.warning(f"✗ {error_msg}")
                for violation in validation['violations']:
                    logging.warning(f"  Violation: {violation}")
            else:
                print(f"   ✗ {error_msg}:")
                for violation in validation['violations']:
                    print(f"     - {violation}")
        
        if validation['warnings']:
            if args.service_mode:
                for warning in validation['warnings']:
                    logging.warning(f"  {warning}")
            else:
                print("   Warnings:")
                for warning in validation['warnings']:
                    print(f"     - {warning}")
        
        # Final completion message
        if args.service_mode:
            logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
        else:
            print("\n" + "="*70)
            print("OPTIMIZATION COMPLETED SUCCESSFULLY")
            print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        interrupt_msg = "Optimization interrupted by user"
        if args.service_mode:
            logging.warning(interrupt_msg)
        else:
            print(f"\n{interrupt_msg}")
        return 1
        
    except Exception as e:
        error_msg = f"Error during optimization: {str(e)}"
        if args.service_mode:
            logging.error(error_msg)
            logging.error("Exception details:", exc_info=True)
        else:
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Import pyomo here to avoid issues with command line parsing
    import pyomo.environ as pyo
    
    exit_code = main()
    sys.exit(exit_code)