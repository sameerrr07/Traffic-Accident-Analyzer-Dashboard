import pandas as pd

# Load data
df = pd.read_csv("data/ADSI_Table_1A.2.csv")

# Clean column names
df.columns = df.columns.str.strip()

# 👉 Automatically detect State column (no hardcoding)
state_col = df.columns[1]

print("STATE COLUMN:", state_col)

print("\nDATA PREVIEW:")
print(df.head())

print("\nINFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# Remove total row
df = df[df[state_col] != "Total"]

print("\nCLEANED DATA:")
print(df.head())

# Top 10 states by accidents
top_states = df.sort_values(
    by="Total Traffic Accidents - Cases", 
    ascending=False
).head(10)

print("\nTOP 10 STATES BY ACCIDENTS:")
print(top_states[[state_col, "Total Traffic Accidents - Cases"]])

# Death rate
df["Death Rate"] = df["Total Traffic Accidents - Died"] / df["Total Traffic Accidents - Cases"]

print("\nDEATH RATE:")
print(df[[state_col, "Death Rate"]].head())

# Most dangerous states
danger_states = df.sort_values(by="Death Rate", ascending=False).head(10)

print("\nMOST DANGEROUS STATES:")
print(danger_states[[state_col, "Death Rate"]])
import matplotlib.pyplot as plt
import numpy as np

top_states = df.sort_values(
    by="Total Traffic Accidents - Cases", 
    ascending=False
).head(10)

# Color gradient create
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_states)))

plt.figure(figsize=(12,6))

bars = plt.bar(
    top_states[state_col],
    top_states["Total Traffic Accidents - Cases"],
    color=colors
)

plt.xticks(rotation=45)
plt.title("🚦 Top 10 Accident States (India)", fontsize=16, weight='bold')
plt.xlabel("State")
plt.ylabel("Number of Accidents")

# Value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval,
             f"{int(yval):,}",
             ha='center', va='bottom', fontsize=9, weight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()