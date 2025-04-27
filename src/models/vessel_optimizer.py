import pulp
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union


class VesselOptimizer:
    def __init__(self, 
                 max_free_voyages: int = 6,
                 vessel_penalty: int = 1000, 
                 slack_penalty: int = 10,
                 lead_time: int = 5,
                 vessel_capacity_1_2_grades: int = 700,
                 vessel_capacity_3_grades: int = 650):
        """
        Initialize the Vessel Optimizer with configurable parameters.
        
        Args:
            max_free_voyages: Maximum number of voyages before penalty
            vessel_penalty: Penalty per voyage beyond max_free_voyages
            slack_penalty: Penalty per day of departure window violation
            lead_time: Discharge + preparation time in days
            vessel_capacity_1_2_grades: Capacity if ≤ 2 grades
            vessel_capacity_3_grades: Capacity if 3 grades
        """
        # Configuration parameters
        self.MAX_FREE_VOYAGES = max_free_voyages
        self.PENALTY_VESSEL = vessel_penalty
        self.PENALTY_SLACK = slack_penalty
        self.LEAD_TIME = lead_time
        self.VEG1_2_CAP = vessel_capacity_1_2_grades 
        self.VEG3_CAP = vessel_capacity_3_grades
        
        # Data containers
        self.parcels = []
        self.grades = []
        self.P = []
        self.vol = {}
        self.ldr_start = {}
        self.ldr_end = {}
        self.origin = {}
        self.transit = {}
        
        # Model components
        self.prob = None
        self.V = []
        self.result = None

    def load_data_from_dict(self, data: Dict) -> None:
        """
        Load data from a dictionary.
        
        Args:
            data: Dictionary containing parcels, transit times, etc.
        """
        self.parcels = data.get('parcels', [])
        self.transit = data.get('transit', {"PM": 2, "Sabah": 3.5, "Sarawak": 1})
        
        # Process parcels data
        self._process_parcels_data()

    def load_data_from_json(self, file_path: str) -> None:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.load_data_from_dict(data)

    def _process_parcels_data(self) -> None:
        """Process the parcels data to create lookup dictionaries."""
        self.grades = sorted({g for _, g, *_ in self.parcels})
        self.P = [p[0] for p in self.parcels]  # parcel IDs
        
        # Quick lookups
        self.vol = {p: v for p, _, v, *_ in self.parcels}
        self.ldr_start = {p: s for p, _, _, s, _, _ in self.parcels}
        self.ldr_end = {p: e for p, _, _, _, e, _ in self.parcels}
        self.origin = {p: o for p, _, _, _, _, o in self.parcels}

    def build_model(self) -> None:
        """Build the MILP optimization model."""
        # Model constants
        self.V_MAX = len(self.parcels)
        self.V = list(range(self.V_MAX))
        self.BIG_M_TIME = 100  # big‑M for window constraints
        self.BIG_M_CAP = sum(self.vol.values())
        
        # Create problem
        self.prob = pulp.LpProblem("Vessel_Scheduling_With_Slack", pulp.LpMinimize)
        
        # Decision vars
        self.x = pulp.LpVariable.dicts("assign", (self.P, self.V), cat="Binary")
        self.y = pulp.LpVariable.dicts("voyage_used", self.V, cat="Binary")
        self.d = pulp.LpVariable.dicts("dep_day", self.V, lowBound=1, upBound=31, cat="Integer")
        self.z = pulp.LpVariable.dicts("has_grade", (self.V, self.grades), cat="Binary")
        
        # Slack variables
        self.s_voy = pulp.LpVariable("slack_voyages", lowBound=0, cat="Integer")
        self.s_early = pulp.LpVariable.dicts("slack_early", self.P, lowBound=0, cat="Continuous")
        self.s_late = pulp.LpVariable.dicts("slack_late", self.P, lowBound=0, cat="Continuous")
        
        # Objective: minimize #voyages + penalties
        self.prob += (
            pulp.lpSum(self.y[v] for v in self.V)
            + self.PENALTY_VESSEL * self.s_voy
            + self.PENALTY_SLACK * pulp.lpSum(self.s_early[p] + self.s_late[p] for p in self.P)
        )
        
        # Add all constraints
        self._add_assignment_constraints()
        self._add_voyage_linking_constraints()
        self._add_grade_indicator_constraints()
        self._add_capacity_constraints()
        self._add_departure_window_constraints()
        self._add_voyage_cap_constraints()

    def _add_assignment_constraints(self) -> None:
        """Add constraints: each parcel on exactly one voyage."""
        for p in self.P:
            self.prob += pulp.lpSum(self.x[p][v] for v in self.V) == 1

    def _add_voyage_linking_constraints(self) -> None:
        """Add constraints: link assign → voyage_used."""
        for v in self.V:
            self.prob += pulp.lpSum(self.x[p][v] for p in self.P) <= self.V_MAX * self.y[v]

    def _add_grade_indicator_constraints(self) -> None:
        """Add grade indicator constraints for piecewise capacity."""
        for v in self.V:
            for g in self.grades:
                # If any parcel p of grade g is on v, then z[v,g]=1
                self.prob += pulp.lpSum(self.x[p][v] for p, pg, *_ in self.parcels if pg == g) >= self.z[v][g]
                for p, pg, *_ in self.parcels:
                    if pg == g:
                        self.prob += self.x[p][v] <= self.z[v][g]

    def _add_capacity_constraints(self) -> None:
        """Add piecewise capacity constraints."""
        for v in self.V:
            total_load = pulp.lpSum(self.vol[p] * self.x[p][v] for p in self.P)
            sum_grades = pulp.lpSum(self.z[v][g] for g in self.grades)
            # always ≤ VEG1_2_CAP if voyage used
            self.prob += total_load <= self.VEG1_2_CAP * self.y[v]
            # if 3 grades, enforce ≤ VEG3_CAP
            self.prob += total_load <= self.VEG3_CAP * self.y[v] + self.BIG_M_CAP * (3 - sum_grades)

    def _add_departure_window_constraints(self) -> None:
        """Add departure window constraints with slack variables."""
        for p in self.P:
            for v in self.V:
                self.prob += (
                    self.d[v]
                    >= self.ldr_start[p] - self.s_early[p]
                    - self.BIG_M_TIME * (1 - self.x[p][v])
                )
                self.prob += (
                    self.d[v]
                    <= self.ldr_end[p] + self.s_late[p]
                    + self.BIG_M_TIME * (1 - self.x[p][v])
                )

    def _add_voyage_cap_constraints(self) -> None:
        """Add soft voyage cap constraints."""
        self.prob += self.s_voy >= pulp.lpSum(self.y[v] for v in self.V) - self.MAX_FREE_VOYAGES

    def get_original_crude(self, parcel_id: str) -> str:
        """
        Get the original crude name from parcel ID.
        
        Args:
            parcel_id: ID of the parcel
            
        Returns:
            Original crude name
        """
        parts = parcel_id.split("_")
        if len(parts) >= 3 and parts[1] in ["E", "F"]:
            return parts[1]  # Return just "E" or "F"
        else:
            return parts[1]  # Return Base, A, B, C, D etc.

    def solve(self, verbose: bool = False) -> Dict:
        """
        Solve the MILP model and return results.
        
        Args:
            verbose: Whether to print solver output
            
        Returns:
            Dictionary containing results
        """
        if self.prob is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
        
        if verbose:
            print("Status:", pulp.LpStatus[self.prob.status])
        
        # Process results
        if self.prob.status == pulp.LpStatusOptimal:
            return self._process_results(verbose)
        else:
            return {
                "status": pulp.LpStatus[self.prob.status],
                "error": "Failed to find optimal solution"
            }

    def solve_multiple(self, num_solutions: int = 3, optimality_gap: float = 0.05, verbose: bool = False) -> List[Dict]:
        """
        Generate multiple different solutions.
        
        Args:
            num_solutions: Number of solutions to generate
            optimality_gap: Maximum gap from optimal (0.05 = 5% worse than optimal)
            verbose: Whether to print solver output
            
        Returns:
            List of solution dictionaries
        """
        if self.prob is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        solutions = []
        
        # Get the first solution
        self.prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
        
        if self.prob.status != pulp.LpStatusOptimal:
            return [{
                "status": pulp.LpStatus[self.prob.status],
                "error": "Failed to find optimal solution"
            }]
        
        # Save the original objective value
        original_obj = pulp.value(self.prob.objective)
        
        # Process and save first solution
        first_solution = self._process_results(verbose)
        solutions.append(first_solution)
        
        # Add constraints to exclude previously found solutions and find more
        for i in range(1, num_solutions):
            # Create integer cut constraint to exclude previous solution
            # Get previous voyage assignments
            prev_solution = solutions[-1]
            used_voyages = [v["voyage_id"] for v in prev_solution["json_data"]["voyages"]]
            
            # Create a constraint that forces at least one change
            # Sum of all previously used y[v] must be <= len(used_voyages) - 1
            # This forces at least one previously used voyage to not be used
            # OR at least one unused voyage to be used
            if used_voyages:
                self.prob += (
                    pulp.lpSum(self.y[v] for v in used_voyages) 
                    <= len(used_voyages) - 1,
                    f"exclude_solution_{i}"
                )
            
                # Add objective bound to ensure we don't get solutions that are too bad
                # This constraint ensures the objective is at most (1+gap) times the original
                max_obj = (1 + optimality_gap) * original_obj
                self.prob += self.prob.objective <= max_obj, f"optimality_gap_{i}"
                
                # Solve with the new constraints
                self.prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
                
                # Check if we found a new solution
                if self.prob.status == pulp.LpStatusOptimal:
                    new_solution = self._process_results(verbose)
                    solutions.append(new_solution)
                else:
                    if verbose:
                        print(f"No more feasible solutions within {optimality_gap*100:.1f}% of optimal")
                    break
            else:
                # No used voyages - can't create more solutions
                break
                
        return solutions

    def _process_results(self, verbose: bool = False) -> Dict:
        """
        Process results from the solved model.
        
        Args:
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary containing processed results
        """
        used = [v for v in self.V if pulp.value(self.y[v]) > 0.5]
        
        if verbose:
            print(f"Voyages used: {len(used)}, slack_voyages = {pulp.value(self.s_voy):.0f}")
        
        # Create data structures for results
        voyages_data = []
        parcels_data = []
        
        for v in used:
            dep = int(pulp.value(self.d[v]))
            assigned = [p for p in self.P if pulp.value(self.x[p][v]) > 0.5]
            total = sum(self.vol[p] for p in assigned)
            grades_v = [g for g in self.grades if pulp.value(self.z[v][g]) > 0.5]
            # approximate arrival:
            arr_day = dep + self.transit[self.origin[assigned[0]]] + self.LEAD_TIME
            
            # Print standard output
            if verbose:
                print(f"\nVoyage {v}: dep_day={dep}, arrival≈{arr_day}, load={total} kb, grades={grades_v}")
            
            # Collect voyage data
            voyage_info = {
                "voyage_id": v,
                "departure_day": dep,
                "approx_arrival": arr_day,
                "total_load": total,
                "grades": grades_v
            }
            voyages_data.append(voyage_info)
            
            for p in assigned:
                e = pulp.value(self.s_early[p])
                l = pulp.value(self.s_late[p])
                
                # Print standard output
                if verbose:
                    print(f"  - {p}: {self.vol[p]} kb, window {self.ldr_start[p]}–{self.ldr_end[p]}, "
                        f"slack_early={e:.1f}, slack_late={l:.1f}")
                
                # Collect parcel data
                parcel_info = {
                    "parcel_id": p,
                    "voyage_id": v,
                    "volume": self.vol[p],
                    "loading_window_start": self.ldr_start[p],
                    "loading_window_end": self.ldr_end[p],
                    "slack_early": e,
                    "slack_late": l,
                    "origin": self.origin[p],
                    "crude": self.get_original_crude(p)
                }
                parcels_data.append(parcel_info)
        
        # Create DataFrames
        voyages_df = pd.DataFrame(voyages_data)
        parcels_df = pd.DataFrame(parcels_data)
        
        # Create summary data
        summary_data = {
            "status": pulp.LpStatus[self.prob.status],
            "voyages_used": len(used),
            "slack_voyages": int(pulp.value(self.s_voy)),
            "objective_value": pulp.value(self.prob.objective)
        }
        summary_df = pd.DataFrame([summary_data])
        
        # --- Generate JSON output ---
        json_data = {
            "status": pulp.LpStatus[self.prob.status],
            "objective_value": pulp.value(self.prob.objective),
            "voyages_used": len(used),
            "slack_voyages": int(pulp.value(self.s_voy)),
            "voyages": voyages_data,
            "parcels": parcels_data
        }
        
        # Convert to JSON string with proper formatting
        json_output = json.dumps(json_data, indent=2)
        
        # Construct result dictionary
        result = {
            "dataframes": {
                "summary": summary_df,
                "voyages": voyages_df, 
                "parcels": parcels_df
            },
            "json_data": json_data
        }
        
        if verbose:
            print("\n--- DataFrame Output ---")
            print("\nSummary:")
            print(summary_df)
            print("\nVoyages:")
            print(voyages_df)
            print("\nParcels:")
            print(parcels_df)
        
        return result
    
    def save_results(self, output_dir: str = '.') -> Dict[str, str]:
        """
        Save results to CSV and JSON files.
        
        Args:
            output_dir: Directory to save files
            
        Returns:
            Dictionary with file paths
        """
        if self.result is None:
            raise ValueError("No results to save. Run solve() first.")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV files
        voyages_path = output_path / "voyages.csv"
        parcels_path = output_path / "parcels.csv"
        summary_path = output_path / "summary.csv"
        
        self.result["dataframes"]["voyages"].to_csv(voyages_path, index=False)
        self.result["dataframes"]["parcels"].to_csv(parcels_path, index=False)
        self.result["dataframes"]["summary"].to_csv(summary_path, index=False)
        
        # Save JSON file
        json_path = output_path / "results.json"
        with open(json_path, "w") as f:
            json.dump(self.result["json_data"], f, indent=2)
        
        return {
            "voyages_csv": str(voyages_path),
            "parcels_csv": str(parcels_path),
            "summary_csv": str(summary_path),
            "results_json": str(json_path)
        }

    def run(self, 
            data: Optional[Dict] = None, 
            json_path: Optional[str] = None,
            save_output: bool = False, 
            output_dir: str = '.',
            verbose: bool = False,
            multiple_solutions: bool = False,
            num_solutions: int = 3,
            optimality_gap: float = 0.05) -> Union[Dict, List[Dict]]:
        """
        Run the full optimization pipeline.
        
        Args:
            data: Optional dictionary with input data
            json_path: Optional path to JSON file with input data
            save_output: Whether to save results to files
            output_dir: Directory to save output files
            verbose: Whether to print detailed output
            multiple_solutions: Whether to generate multiple solutions
            num_solutions: Number of solutions to generate if multiple_solutions=True
            optimality_gap: Maximum gap from optimal solution (0.05 = 5%)
            
        Returns:
            Dictionary with optimization results, or list of dictionaries if multiple_solutions=True
        """
        # Load data
        if data is not None:
            self.load_data_from_dict(data)
        elif json_path is not None:
            self.load_data_from_json(json_path)
        elif not self.parcels:  # No data loaded yet
            raise ValueError("No data provided. Pass 'data' or 'json_path'.")
        
        # Build model
        self.build_model()
        
        # Generate solution(s)
        if multiple_solutions:
            results = self.solve_multiple(num_solutions, optimality_gap, verbose)
            
            # Save results if requested
            if save_output:
                for i, result in enumerate(results):
                    # Create a subdirectory for each solution
                    solution_dir = Path(output_dir) / f"solution_{i+1}"
                    solution_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Temporarily set self.result to the current solution
                    self.result = result
                    file_paths = self.save_results(str(solution_dir))
                    result["files"] = file_paths
                
            return results
        else:
            # Single solution
            self.result = self.solve(verbose=verbose)
            
            # Save result if requested
            if save_output:
                file_paths = self.save_results(output_dir)
                self.result["files"] = file_paths
            
            return self.result


# Example usage when run as a script
if __name__ == "__main__":
    # Sample data
    sample_data = {
        "parcels": [
            ("P_Base_1", "Base", 400, 1, 3, "PM"),
            ("P_Base_2", "Base", 360, 17, 19, "PM"),
            ("P_A", "A", 150, 1, 3, "PM"),
            ("P_B", "B", 150, 8, 10, "PM"),
            ("P_C", "C", 200, 17, 19, "PM"),
            ("P_D", "D", 150, 23, 24, "Sabah"),
            ("P_E1", "E", 250, 5, 7, "Sabah"),
            ("P_E2", "E", 550, 10, 11, "Sabah"),
            ("P_E3", "E", 100, 23, 24, "Sabah"),
            ("P_F1", "F", 320, 5, 7, "Sarawak"),
            ("P_F2", "F", 280, 19, 22, "Sarawak")
        ],
        "transit": {"PM": 2, "Sabah": 3.5, "Sarawak": 1}
    }
    
    # Create optimizer with custom parameters
    optimizer = VesselOptimizer(
        max_free_voyages=5,
        vessel_penalty=1000,
        slack_penalty=10,
        lead_time=5,  # discharge + prep
        vessel_capacity_1_2_grades=700,
        vessel_capacity_3_grades=650
    )
    
    # Run single solution example
    results_single = optimizer.run(
        data=sample_data,
        save_output=True,
        verbose=True
    )

    print("Single solution optimization completed successfully!")
    print(f"  Objective value: {results_single['json_data']['objective_value']}")
    print(f"  Voyages used: {results_single['json_data']['voyages_used']}")

    # Multiple solutions example
    results_multiple = optimizer.run(
        data=sample_data,
        save_output=True,
        verbose=True,
        multiple_solutions=True,  # Set to True to get multiple solutions
        num_solutions=3,
        optimality_gap=0.05  # Allow solutions up to 5% worse than optimal
    )

    print("\nMultiple solutions optimization completed successfully!")
    for i, solution in enumerate(results_multiple):
        print(f"\nSolution {i+1}:")
        print(f"  Objective value: {solution['json_data']['objective_value']}")
        print(f"  Voyages used: {solution['json_data']['voyages_used']}")
        
        # Print voyages
        for voyage in solution["json_data"]["voyages"]:
            print(f"  - Voyage {voyage['voyage_id']}: departure={voyage['departure_day']}, "
                  f"arrival≈{voyage['approx_arrival']}")