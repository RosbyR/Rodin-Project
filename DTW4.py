import pandas as pd
from dtaidistance import dtw
import matplotlib.pyplot as plt
from dtaidistance import dtw_visualisation as dtwvis
from scipy import stats
from dtaidistance.preprocessing import differencing

# List of file paths
file_paths = [
    'correct data/1324_20230822_092316_seg_3_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_101847_seg_3_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_111723_seg_2_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_115811_seg_3_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_125047_seg_2_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_132720_seg_3_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_140717_seg_3_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_144911_seg_3_Stand_cycles_ensemble.csv',
    'correct data/1324_20230822_152936_seg_3_Stand_cycles_ensemble.csv',  
]

# Load the baseline dataset (first file in the list)
baseline_df = pd.read_csv(file_paths[0], header=None)
baseline_sequence = baseline_df.iloc[0].values  # Extract the baseline sequence
# Z-normalize the baseline
baseline_sequence = stats.zscore(baseline_sequence)

# Optionally apply differencing to baseline
baseline_sequence = differencing(baseline_sequence, smooth=0.1)

# Iterate over the rest of the files and compare them to the baseline
for file_path in file_paths[1:]:
    # Load the current file
    input_df = pd.read_csv(file_path, header=None)
    input_sequence = input_df.iloc[0].values  # Extract the sequence
    
    # Z-normalize the input sequence
    input_sequence = stats.zscore(input_sequence)
    
    # Optionally apply differencing to input sequence
    input_sequence = differencing(input_sequence, smooth=0.1)
    
    # Calculate the DTW alignment cost
    alignment_cost = dtw.distance(baseline_sequence, input_sequence)
    print(f"DTW Alignment Cost for {file_path.split('/')[-1]}: {alignment_cost}")
    
    # Visualize the alignment
    alignment_path = dtw.warping_path
