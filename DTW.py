import pandas as pd
from dtaidistance import dtw
from scipy import stats

# Load the dataset
file_path = "Processed_Cycles_Stand.xlsx"
df = pd.read_excel(file_path, header=0)  # First row as headers

# Extract unique patient IDs (Subject ID in first column)
patient_ids = df.iloc[:, 0].unique()

dtw_results = []  # Store results for CSV output

for patient_id in patient_ids:
    patient_data = df[df.iloc[:, 0] == patient_id].reset_index(drop=True)
    
    if patient_data.shape[0] > 1:  # Ensure there are comparisons to be made
        numeric_data = patient_data.iloc[:, 7:].apply(pd.to_numeric, errors='coerce')  # Extract numerical data
        numeric_data = numeric_data.dropna(axis=1, how='all')  # Drop entirely non-numeric columns
        
        if numeric_data.shape[1] == 0:
            continue  # Skip if no numeric data is available
        
        baseline_sequence = stats.zscore(numeric_data.iloc[0].values)  # First row as baseline
        
        for idx in range(1, numeric_data.shape[0]):
            comparison_sequence = stats.zscore(numeric_data.iloc[idx].values)
            
            if len(baseline_sequence) == len(comparison_sequence):  # Ensure sequences match in length
                alignment_cost = dtw.distance(baseline_sequence, comparison_sequence)
                dtw_results.append([patient_id, idx, alignment_cost])

# Save results to CSV
output_path = "DTW_Costs.csv"
dtw_df = pd.DataFrame(dtw_results, columns=["Patient_ID", "Comparison_Index", "DTW_Cost"])
dtw_df.to_csv(output_path, index=False)

print(f"DTW cost analysis completed. Results saved to {output_path}")
