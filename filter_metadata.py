import pandas as pd
import os

# Create dataset_copy directory if it doesn't exist
os.makedirs('dataset_copy', exist_ok=True)

# Load metadata
metadata = pd.read_csv('images.csv')

# Define columns to include
columns_to_include = ['imgPath', 'caption', 'chartType', 'xAxis', 'yAxis']

# Filter the metadata
filtered_metadata = metadata[columns_to_include]

# Save the filtered metadata
filtered_metadata.to_csv('dataset_copy/filtered_metadata.csv', index=False)
print("Filtered metadata saved as 'filtered_metadata.csv'.")
