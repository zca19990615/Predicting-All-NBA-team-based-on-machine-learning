# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:10:58 2024

@author: 61967
"""

import matplotlib.pyplot as plt
import numpy as np

# Data: Sensitivity values for different models
models = ['Naïve Bayes', 'KNN', 'LR-EN', 'GBM', 'SVC']
sensitivity = [0.9298, 0.9959, 0.7438, 0.8802, 0.7851]

# Corrected error range: Lower and upper bounds for each model (matching the provided image)
error_lower = [0.8947, 0.9869, 0.6903, 0.8802, 0.7293]
error_upper = [0.9603, 0.9998, 0.7976, 0.9188, 0.8341]

# Calculate the asymmetric error bars
error = [
    [sensitivity[i] - error_lower[i], error_upper[i] - sensitivity[i]]
    for i in range(len(sensitivity))
]

# Convert error to a format suitable for matplotlib (transpose to match yerr format)
asymmetric_error = np.array(error).T

# Create bar chart
plt.figure(figsize=(8, 6))
colors = ['#6ec6ca', '#f15b50', '#45b39d', '#354a7c', '#f28b74']  # Matching the colors from the provided image
bar_width = 0.6  # Adjusting the bar width to match the chart
bars = plt.bar(models, sensitivity, color=colors, width=bar_width, yerr=asymmetric_error, capsize=5, ecolor='gray', alpha=1.0)  # Keep bars opaque, use asymmetric error bars

# Add title and labels
plt.title('TrainSet', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)

# Adjust y-axis ticks
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Add horizontal grid lines and remove vertical grid lines
plt.grid(axis='y', linestyle='--')  # Add horizontal grid lines
plt.grid(axis='x', visible=False)  # Hide vertical grid lines

# Add legend to the right side of the plot, displaying color and sensitivity score with error
plt.legend(bars, [f"{score:.4f} ({lower:.4f}-{upper:.4f})" for score, lower, upper in zip(sensitivity, error_lower, error_upper)], loc='center left', fontsize=10, title='Model Sensitivity Score', title_fontsize=12, frameon=True, bbox_to_anchor=(1, 0.5))

# Display the chart
plt.tight_layout()
plt.show()
