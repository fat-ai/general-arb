import pickle
from pathlib import Path
from sim_strat_3 import BayesianState # Needed so pickle recognizes the class

CACHE_DIR = Path("./data-cache/polymarket_cache")
SIM_FILE = CACHE_DIR / "sim_checkpoint.pkl"
LIVE_FILE = CACHE_DIR / "bayesian_state.pkl"

def handoff_state():
    print("Loading simulation checkpoint...")
    with open(SIM_FILE, 'rb') as f:
        checkpoint_data = pickle.load(f)
        
    # Extract just the BayesianState from the dictionary
    mature_state = checkpoint_data['state']
    
    print(f"Extracted state. Last processed timestamp: {mature_state.last_processed_timestamp}")
    
    print("Saving for live daily updater...")
    with open(LIVE_FILE, 'wb') as f:
        pickle.dump(mature_state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("✅ Handoff complete! daily_update.py is ready to run.")

if __name__ == "__main__":
    handoff_state()
