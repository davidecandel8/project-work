# Import Student Solution and Problem Definition
import os
from s347245 import solution
from Problem import Problem

# --- TEST SUITE ENGINE ---

def run_test_suite():
    """
    Executes the comprehensive test suite based on the user's requested configurations.
    Reports are saved in './test_reports/'.
    """
    
    # 1. Define Test Cases (Tuple: N, Density, Alpha, Beta)
    # Based on the user's ic(...) requests
    test_cases = [
        (100, 0.2, 1, 1),
        (100, 0.2, 2, 1),
        (100, 0.2, 1, 2),
        (100, 1, 1, 1),
        (100, 1, 2, 1),
        (100, 1, 1, 2),
        (1000, 0.2, 1, 1),
        (1000, 0.2, 2, 1),
        (1000, 0.2, 1, 2),
        (1000, 1, 1, 1),
        (1000, 1, 2, 1),
        (1000, 1, 1, 2)
    ]
    
    base_folder = "test_reports"
    os.makedirs(base_folder, exist_ok=True)
    
    print("="*60)
    print("      COMPUTATIONAL INTELLIGENCE - AUTOMATED TEST SUITE      ")
    print("="*60)
    print(f"Detected {len(test_cases)} configurations to run.")
    print(f"Reports will be saved in: ./{base_folder}/")
    print("="*60 + "\n")
    
    for i, (n, dens, a, b) in enumerate(test_cases):
        print(f"\n>>> TEST RUN {i+1}/{len(test_cases)}: Size={n}, Density={dens}, Alpha={a}, Beta={b}")
        
        # Instantiate Problem
        # Note: 'seed' fixed to 42 for reproducibility as per original file usage
        p = Problem(num_cities=n, density=dens, alpha=a, beta=b, seed=42)
        
        # Generate dynamic filename
        filename = f"report_N{n}_D{dens}_A{a}_B{b}.txt"
        filepath = os.path.join(base_folder, filename)
        
        # Run Solver
        solution(p, report_path=filepath)
        
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY.")
    print("="*60)

if __name__ == "__main__":
    # Just run the suite
    run_test_suite()