# Rodin-Project
Dynamic Time Warping (DTW)
Dynamic Time Warping (DTW) measures the similarity between two time-series signals by aligning them non-linearly, handling differences in timing or speed. It calculates a DTW cost, representing the difference between the signals based on an optimal alignment path through a cost matrix.

How it Works:

Compute a distance matrix for all signal points (e.g., Euclidean distance).
Use cumulative cost to find the optimal alignment path.
The total cost of this path is the DTW cost.
Key Insights:

Low Cost: Signals are similar, even with timing differences.
High Cost: Signals differ significantly in shape or structure.
DTW is widely used in signal processing, speech recognition, and gesture analysis due to its ability to handle temporal misalignments.







