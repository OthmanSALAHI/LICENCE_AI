import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 10,000 rows
n = 10000

# Player_Age: Normally distributed between 18-39
ages = np.clip(np.random.normal(loc=28, scale=5, size=n).astype(int), 18, 39)

# Player_Weight: Normally distributed between 40-105 (mean=75, std=10)
weights = np.round(np.clip(np.random.normal(loc=75, scale=10, size=n), 40, 105), 5)

# Player_Height: Normally distributed between 150-210 (mean=180, std=10)
heights = np.round(np.clip(np.random.normal(loc=180, scale=10, size=n), 150, 210), 5)

# Previous_Injuries: Binary (0 or 1)
previous_injuries = np.random.binomial(1, p=0.5, size=n)

# Training_Intensity: Uniform distribution between 0-1
training_intensity = np.round(np.random.uniform(0, 1, size=n), 5)

# Recovery_Time: Integers between 1-6
recovery_time = np.random.randint(1, 7, size=n)

# Likelihood_of_Injury: Binary (0 or 1)
likelihood_of_injury = np.random.binomial(1, p=0.5, size=n)

# Create DataFrame
df = pd.DataFrame({
    'Player_Age': ages,
    'Player_Weight': weights,
    'Player_Height': heights,
    'Previous_Injuries': previous_injuries,
    'Training_Intensity': training_intensity,
    'Recovery_Time': recovery_time,
    'Likelihood_of_Injury': likelihood_of_injury
})

# Save to CSV
df.to_csv('injury_data.csv', index=False)