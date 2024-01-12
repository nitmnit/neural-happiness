import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample data with varying variance
data1 = np.random.normal(loc=50, scale=10, size=100)  # Lower variance
data2 = np.random.normal(loc=50, scale=20, size=100)  # Higher variance

# Create a pandas DataFrame
df = pd.DataFrame({"data1": data1, "data2": data2})

# Calculate variance using pandas
variance1 = df["data1"].var()
variance2 = df["data2"].var()

# Create box plots to visualize variance
plt.figure(figsize=(8, 6))
plt.boxplot([data1, data2], labels=["Lower Variance", "Higher Variance"])
plt.title("Box Plots Visualizing Variance")
plt.show()

# Create violin plots for a more detailed view
plt.figure(figsize=(8, 6))
plt.violinplot([data1, data2], showmeans=True, showmedians=True)
plt.title("Violin Plots Showing Density and Spread")
plt.show()
