import pandas as pd
import os

print("=== TRANSITION ANALYSIS ===")
print("")

# Find the most recent result files
result_files = [f for f in os.listdir("results/") if f.startswith("crude_blending_") and f.endswith(".csv")]
result_files.sort(key=lambda x: os.path.getmtime(f"results/{x}"), reverse=True)

if not result_files:
    print("No result files found!")
    exit()

latest_file = result_files[0]
print(f"Analyzing: {latest_file}")

# Load the results
df = pd.read_csv(f"results/{latest_file}")

# Count transitions between different final products
transitions = 0
prev_product = None

transition_details = []
for _, row in df.iterrows():
    current_product = row['Final Product']
    if prev_product is not None and prev_product != current_product:
        transitions += 1
        transition_details.append(f"Day {row['Date']}, Slot {row['Slot']}: {prev_product} -> {current_product}")
    prev_product = current_product

print(f"\nTotal Transitions: {transitions}")
print(f"MaxTransitions limit from config: 11")
print(f"Constraint violated: {'YES' if transitions > 11 else 'NO'}")

if transitions <= 15:  # Only show details if not too many
    print("\n=== TRANSITION DETAILS ===")
    for detail in transition_details:
        print(detail)

print(f"\n=== PRODUCT SEQUENCE ===")
sequence = ' -> '.join(df['Final Product'].tolist())
if len(sequence) > 200:
    print("Product sequence (first 200 chars):", sequence[:200] + "...")
else:
    print("Product sequence:", sequence)
