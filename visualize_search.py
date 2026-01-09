"""
Visualize Architecture Search Results
Bubble plot of Accuracy vs (Qubits, Depth, Topology)
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    with open("results/architecture_search.json", "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Create bubble plot
    plt.figure(figsize=(12, 8))
    
    # Map topology to color
    topo_colors = {"chain": "steelblue", "ring": "coral", "all2all": "purple"}
    colors = df["topology"].map(topo_colors)
    
    # Map depth to size
    sizes = df["depth"] * 100 + 100
    
    # Jitter x-axis (qubits) slightly for visibility
    x_jitter = df["qubits"] + (pd.Series([0, 1, 2] * (len(df)//3))[:len(df)] - 1) * 0.3
    
    plt.scatter(x_jitter, df["accuracy"], s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=2)
    
    # Add labels
    for i, row in df.iterrows():
        plt.text(x_jitter[i], row["accuracy"] + 0.01, f"D{row['depth']}", ha='center', fontsize=8)
    
    # Legend
    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in topo_colors.values()]
    plt.legend(markers, topo_colors.keys(), title="Topology")
    
    plt.xlabel("Number of Qubits")
    plt.ylabel("Test Accuracy")
    plt.title("Architecture Search: Circle Task (Size = Depth)")
    plt.xticks([4, 8])
    plt.grid(True, alpha=0.3)
    
    # Add reference line for Classical baseline (0.69)
    plt.axhline(0.69, color='red', linestyle='--', label='Classical Baseline')
    plt.text(4.1, 0.695, "Classical MLP (0.69)", color='red')
    
    plt.tight_layout()
    plt.savefig("results/architecture_search_bubble.png", dpi=150)
    print("Saved: results/architecture_search_bubble.png")

if __name__ == "__main__":
    main()
