import pandas as pd
import numpy as np

def generate_data():
    """
    Generates hypothetical data for 5 3PL providers and 6 criteria across 3 scenarios.
    
    Providers: 3PL-A, 3PL-B, 3PL-C, 3PL-D, 3PL-E
    Criteria: Cost, Time, Flexibility, Reliability, Capacity, Coverage
    Scenarios: Normal, Peak Demand, Disruption
    
    Returns:
        dictionary of pandas DataFrames for each scenario
    """
    np.random.seed(42) # For reproducibility
    
    providers = ['3PL-A', '3PL-B', '3PL-C', '3PL-D', '3PL-E']
    criteria = ['Cost', 'Time', 'Flexibility', 'Reliability', 'Capacity', 'Coverage']
    
    # Define base ranges for random generation to simulate different provider strengths
    # This ensures the data isn't completely random noise and has some structure
    
    # Scenario 1: Normal Operation
    # Random values between 50 and 90
    data_normal = np.random.randint(50, 95, size=(5, 6))
    df_normal = pd.DataFrame(data_normal, index=providers, columns=criteria)
    
    # Scenario 2: Peak Demand
    # Capacity and Time might suffer for some, Flexibility becomes crucial
    data_peak = data_normal.copy()
    # Simulate stress: Reduce Reliability and Time scores for some providers
    data_peak[0, 1] -= 10 # 3PL-A Time worsens
    data_peak[1, 3] -= 15 # 3PL-B Reliability drops
    # Boost Flexibility importance implicitly by having variance there? 
    # For now just modifying values to show change
    data_peak[2, 2] += 5  # 3PL-C Flexibility shines
    df_peak = pd.DataFrame(data_peak, index=providers, columns=criteria)
    
    # Scenario 3: Disruption
    # Major hits to Coverage and Cost
    data_disruption = data_normal.copy()
    data_disruption[:, 0] -= 20 # Costs increase (score drops if higher is better? Usually Cost is lower is better, but for scoring we often invert or normalize. Let's assume 0-100 Score where 100 is BEST).
    # ASSUMPTION: All scores are "Higher is Better". So for Cost, 100 = Cheapest, 0 = Most Expensive.
    
    data_disruption[3, 5] -= 30 # 3PL-D Coverage hit hard
    data_disruption[4, 2] += 10 # 3PL-E adapts well (Flexibility)
    
    df_disruption = pd.DataFrame(data_disruption, index=providers, columns=criteria)
    
    return {
        'Normal': df_normal,
        'Peak': df_peak,
        'Disruption': df_disruption
    }

if __name__ == "__main__":
    datasets = generate_data()
    for name, df in datasets.items():
        print(f"--- {name} Scenario ---")
        print(df)
        print("\n")
