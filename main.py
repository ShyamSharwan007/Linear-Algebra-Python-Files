import pandas as pd
import matplotlib.pyplot as plt
from data_generator import generate_data
from normalizer import normalize_data
from scorer import calculate_scores
from analyzer import perform_analysis

def main():
    print("Starting Linear Algebra 3PL Evaluation Project...\n")
    
    # 1. Generate Data
    print("Step 1: Generating Dataset...")
    datasets = generate_data()
    
    # Store rankings for final comparison
    all_rankings = {}

    for scenario, df in datasets.items():
        print(f"\n{'='*20} {scenario} Scenario {'='*20}")
        print("Raw Data:")
        print(df)
        
        # 2. Normalize
        print(f"\nStep 2: Normalizing Data ({scenario})...")
        df_minmax, df_zscore = normalize_data(df)
        
        # 3. Score & Rank
        print(f"\nStep 3: Calculating Scores & Ranking ({scenario})...")
        results = calculate_scores(df_minmax)
        print("Rankings (based on weighted score):")
        print(results[['Final Score', 'Rank']])
        
        # Store for comparison
        all_rankings[scenario] = results['Rank']
        
        # 4. PCA & SVD
        print(f"\nStep 4: Performing PCA & SVD ({scenario})...")
        analysis_results = perform_analysis(df_zscore, scenario)
        
        print(f"PCA Explained Variance: {analysis_results['explained_variance']}")
        print(f"SVD Singular Values: {analysis_results['S']}")
        print(f"SVD Denoised Data (Rank-2 Approximation):\n{analysis_results['reconstructed_data'].round(2)}")
        print(f"Plot saved as {scenario}_pca_plot.png")

    # 5. Scenario Analysis Summary
    print(f"\n{'='*20} Final Scenario Analysis {'='*20}")
    comparison_df = pd.DataFrame(all_rankings)
    comparison_df['Average Rank'] = comparison_df.mean(axis=1)
    comparison_df['Stability (Std Dev)'] = comparison_df.std(axis=1)
    
    print("Rankings across all scenarios:")
    print(comparison_df.sort_values('Average Rank'))
    
    best_provider = comparison_df['Average Rank'].idxmin()
    most_stable = comparison_df['Stability (Std Dev)'].idxmin()
    
    print(f"\nBest Overall Provider: {best_provider}")
    print(f"Most Stable Provider: {most_stable}")

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
