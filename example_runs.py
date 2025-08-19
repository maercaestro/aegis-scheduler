"""
Example script showing how to run the optimization with different parameters.
"""

from main import OptimizationRunner
import config

def run_margin_optimization():
    """Run optimization with margin objective."""
    print("Running margin optimization...")
    
    runner = OptimizationRunner(
        data_path=config.DATA_PATH,
        output_path=config.OUTPUT_PATH
    )
    
    results = runner.run_optimization(
        vessel_count=config.VESSEL_COUNT,
        optimization_type="margin",
        scenario_name=f"{config.SCENARIO_NAME}_margin"
    )
    
    return results

def run_throughput_optimization():
    """Run optimization with throughput objective."""
    print("Running throughput optimization...")
    
    runner = OptimizationRunner(
        data_path=config.DATA_PATH,
        output_path=config.OUTPUT_PATH
    )
    
    results = runner.run_optimization(
        vessel_count=config.VESSEL_COUNT,
        optimization_type="throughput",
        max_demurrage_limit=config.MAX_DEMURRAGE_LIMIT,
        scenario_name=f"{config.SCENARIO_NAME}_throughput"
    )
    
    return results

def run_vessel_count_analysis():
    """Run optimization with different vessel counts."""
    print("Running vessel count analysis...")
    
    runner = OptimizationRunner(
        data_path=config.DATA_PATH,
        output_path=config.OUTPUT_PATH
    )
    
    vessel_counts = [4, 5, 6, 7, 8]
    results = {}
    
    for vessel_count in vessel_counts:
        print(f"\nOptimizing with {vessel_count} vessels...")
        result = runner.run_optimization(
            vessel_count=vessel_count,
            optimization_type=config.OPTIMIZATION_TYPE,
            max_demurrage_limit=config.MAX_DEMURRAGE_LIMIT,
            scenario_name=f"{config.SCENARIO_NAME}_{vessel_count}vessels"
        )
        results[vessel_count] = result
        
        if result["status"] == "success":
            metrics = result["summary_metrics"]
            print(f"  Throughput: {metrics['total_throughput']:.1f} k barrels")
            print(f"  Margin: ${metrics['total_margin']:,.0f}")
            print(f"  Demurrage: ${metrics['total_demurrage']:,.0f}")
    
    return results

if __name__ == "__main__":
    print("=== Aegis Scheduler - Example Runs ===\n")
    
    # Uncomment the optimization type you want to run:
    
    # 1. Run margin optimization
    # run_margin_optimization()
    
    # 2. Run throughput optimization
    run_throughput_optimization()
    
    # 3. Run vessel count analysis
    # run_vessel_count_analysis()
