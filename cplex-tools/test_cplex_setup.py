#!/usr/bin/env python3
"""
Test script to verify CPLEX setup and basic functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import cplex
        print(f"✓ CPLEX Python API v{cplex.__version__}")
    except ImportError as e:
        print(f"✗ CPLEX Python API: {e}")
        return False
    
    try:
        import pyomo.environ as pyo
        print("✓ Pyomo")
    except ImportError as e:
        print(f"✗ Pyomo: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas v{pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        return False
    
    return True

def test_solver():
    """Test CPLEX solver availability"""
    print("\nTesting CPLEX solver...")
    try:
        from pyomo.environ import SolverFactory
        
        # Test direct interface
        solver = SolverFactory('cplex_direct')
        if solver.available():
            print("✓ CPLEX Direct interface available")
            solver.options['timelimit'] = 10
            print("✓ Solver options can be set")
            return True
        else:
            print("✗ CPLEX Direct interface not available")
            
        # Fallback to standard interface
        solver = SolverFactory('cplex')
        if solver.available():
            print("✓ CPLEX standard interface available")
            return True
        else:
            print("✗ CPLEX standard interface not available")
            
    except Exception as e:
        print(f"✗ Solver test failed: {e}")
        return False
    
    return False

def test_simple_model():
    """Test with a simple optimization model"""
    print("\nTesting simple optimization model...")
    try:
        from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, maximize, value, SolverFactory, TerminationCondition
        
        # Create a simple model
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals)
        model.y = Var(domain=NonNegativeReals)
        
        model.constraint1 = Constraint(expr=model.x + 2*model.y <= 10)
        model.constraint2 = Constraint(expr=2*model.x + model.y <= 10)
        
        model.objective = Objective(expr=model.x + model.y, sense=maximize)
        
        # Solve with CPLEX
        solver = SolverFactory('cplex_direct')
        if not solver.available():
            solver = SolverFactory('cplex')
            
        if solver.available():
            results = solver.solve(model, tee=False)
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                print("✓ Simple model solved successfully")
                print(f"  Optimal value: {value(model.objective):.2f}")
                return True
            else:
                print("✗ Model solve failed - not optimal")
                return False
        else:
            print("✗ No CPLEX solver available for testing")
            return False
            
    except Exception as e:
        print(f"✗ Simple model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CPLEX Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_solver()
    all_passed &= test_simple_model()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! CPLEX setup is working correctly.")
        print("\nYou can now run the refinery optimization script:")
        print("python cplex_refinery_optimizer.py")
    else:
        print("✗ Some tests failed. Please check your CPLEX installation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
