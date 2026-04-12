import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_correlations():
    x_path = "Data/X_train.npy"
    y_path = "Data/y_train.npy"
    names_path = "Data/feature_names.txt"
    output_dir = "TuningResults"
    
    if not os.path.exists(x_path):
        print("ERROR: Dataset not found. Run Step [4] first!")
        return

    # 1. Load data
    X = np.load(x_path)
    y = np.load(y_path)
    
    with open(names_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    # 2. Extract the "Most Recent" swing (Step 5) for analysis 
    # (This is the swing that most directly influences the future target)
    X_last_step = X[:, -1, :] 
    
    # 3. Create DataFrame
    df = pd.DataFrame(X_last_step, columns=feature_names)
    df['TARGET_RETURN'] = y

    # 4. Calculate Correlation with Target
    corrs = df.corr()[['TARGET_RETURN']].sort_values(by='TARGET_RETURN', ascending=False)

    print("=" * 60)
    print("SwingLab: Feature Correlation Leaderboard")
    print("=" * 60)
    print(corrs)
    print("-" * 60)

    # 5. Create Heatmap
    plt.figure(figsize=(12, 16))
    sns.heatmap(corrs, annot=True, cmap='RdBu_r', center=0, vmin=-0.2, vmax=0.2)
    plt.title("Correlation: Features vs. Future Target\n(Step 5 of 5 Lookback)", fontsize=18, fontweight='bold')
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "Feature_Correlation_Heatmap.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    print(f"SUCCESS: Heatmap saved to: {plot_path}")
    print("=" * 60)
    plt.show()

if __name__ == "__main__":
    analyze_correlations()
