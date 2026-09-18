import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime
import time

# ==============================================================================
# 1. THE 401 PREDICT ENGINE (YOUR CORE MATH)
# ==============================================================================

def normalize_to_10_scale(val, min_val=0.0, max_val=100.0, reverse=False):
    if max_val == min_val: return 5.5
    scaled = 1.0 + ((val - min_val) / (max_val - min_val)) * 9.0
    if reverse: scaled = 11.0 - scaled
    return float(np.clip(scaled, 1.0, 10.0))

def compute_weekly_coefficients(team_stats_dict):
    """
    Computes V1-V5 on the strict 1.0 to 10.0 scale.
    Note: In a real scenario, these stats are pulled from the team's last 5 matches.
    """
    raw_v1 = (team_stats_dict.get("tackles_won_pct", 50.0) + team_stats_dict.get("interception_share_pct", 50.0)) / 2.0
    v1 = normalize_to_10_scale(raw_v1, 0.0, 100.0)

    total_duels = team_stats_dict.get("total_team_duels", 25)
    v2 = 5.5 if total_duels < 20 else normalize_to_10_scale((team_stats_dict.get("ground_duels_won_pct", 50.0) + team_stats_dict.get("aerial_duels_won_pct", 50.0)) / 2.0, 0.0, 100.0)

    v3 = ((team_stats_dict.get("save_pct", 70.0) / 100.0) * 0.7 * 9.0) + ((team_stats_dict.get("keeper_match_rating", 6.0) / 10.0) * 0.3 * 9.0) + 1.0
    
    total_shots = team_stats_dict.get("total_shots", 10)
    if total_shots == 0:
        v4 = 1.0
    else:
        raw_v4 = ((team_stats_dict.get("shots_on_target", 3)/total_shots)*100*0.5) + ((team_stats_dict.get("big_chances", 1)/total_shots)*100*0.5)
        v4 = normalize_to_10_scale(raw_v4, 0.0, 100.0)

    v5 = float(np.clip(team_stats_dict.get("average_team_performance_rating", 6.5), 1.0, 10.0))
    cw = (v1 + v2 + v3 + v4 + v5) / 5.0
    return round(cw, 2)

def calculate_comprehensive_probabilities(home_stats, away_stats):
    # Core Poisson Lambdas (xG Based)
    lh = home_stats.get("xG", 1.5)
    la = away_stats.get("xG", 1.2)

    p_home = [poisson.pmf(i, lh) for i in range(7)]
    p_away = [poisson.pmf(i, la) for i in range(7)]

    # Probabilities Calculation
    win_h = sum(p_home[h] * sum(p_away[:h]) for h in range(1, 7)) * 100
    draw = sum(p_home[i] * p_away[i] for i in range(7)) * 100
    win_a = sum(p_away[a] * sum(p_home[:a]) for a in range(1, 7)) * 100
    
    ov15 = (1 - (p_home[0]*p_away[0]) - (p_home[1]*p_away[0]) - (p_home[0]*p_away[1])) * 100
    ov25 = (1 - sum(p_home[h]*p_away[a] for h in range(7) for a in range(7) if h+a <= 2)) * 100
    btts = (1 - p_home[0]) * (1 - p_away[0]) * 100
    
    # Generate 15 outcome dictionary
    outcomes = {
        "Home Win": win_h, "Away Win": win_a, "Draw": draw,
        "Over 1.5 Goals": ov15, "Over 2.5 Goals": ov25, "Under 2.5 Goals": 100 - ov25,
        "BTTS Yes": btts, "BTTS No": 100 - btts,
        "Double Chance 1X": win_h + draw, "Double Chance X2": win_a + draw,
        "HT Over 0.5": (1 - np.exp(-(lh * 0.45 + la * 0.45))) * 100,
        "Home Clean Sheet": p_away[0] * 100, "Away Clean Sheet": p_home[0] * 100,
        "Home Win To Nil": (sum(p_home[1:]) * p_away[0]) * 100,
        "Over 3.5 Goals": (1 - sum(p_home[h]*p_away[a] for h in range(7) for a in range(7) if h+a <= 3)) * 100
    }
    return outcomes

# ==============================================================================
# 2. DATA ACQUISITION LAYER (API)
# ==============================================================================

class FootballDataClient:
    def __init__(self, api_key):
        self.headers = {'X-Auth-Token': api_key}
        self.base_url = "https://api.football-data.org/v4"

    def get_fixtures(self):
        """Fetch today's matches from major leagues."""
        endpoint = f"{self.base_url}/matches"
        response = requests.get(endpoint, headers=self.headers)
        if response.status_code == 200:
            return response.json().get('matches', [])
        return []

    def mock_team_stats(self, team_name):
        """
        Since free APIs don't provide deep technical stats (tackles/duels) for future games,
        this function simulates the 401 Engine inputs based on team strength.
        """
        # In a production environment, you would query a historical database here.
        np.random.seed(len(team_name))
        return {
            "tackles_won_pct": np.random.uniform(50, 70),
            "interception_share_pct": np.random.uniform(40, 60),
            "total_team_duels": 50,
            "ground_duels_won_pct": np.random.uniform(45, 55),
            "aerial_duels_won_pct": np.random.uniform(45, 55),
            "save_pct": np.random.uniform(65, 80),
            "keeper_match_rating": np.random.uniform(6.0, 8.0),
            "total_shots": 12, "shots_on_target": 4, "big_chances": 2,
            "average_team_performance_rating": np.random.uniform(6.2, 7.5),
            "xG": np.random.uniform(1.0, 2.5)
        }

# ==============================================================================
# 3. EXECUTION & SORTING LOGIC
# ==============================================================================

def main():
    API_KEY = "YOUR_FREE_API_KEY_HERE" # Get one at football-data.org
    client = FootballDataClient(API_KEY)
    
    print(f"--- 401 PREDICT TOOL REPORT: {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")
    
    fixtures = client.get_fixtures()
    if not fixtures:
        print("No matches found for today or API key invalid.")
        return

    for match in fixtures:
        home_name = match['homeTeam']['name']
        away_name = match['awayTeam']['name']
        league = match['competition']['name']

        # Generate Engine Inputs
        h_stats = client.mock_team_stats(home_name)
        a_stats = client.mock_team_stats(away_name)

        # Calculate C_W
        cw_h = compute_weekly_coefficients(h_stats)
        cw_a = compute_weekly_coefficients(a_stats)

        # Calculate 15 Outcomes
        results = calculate_comprehensive_probabilities(h_stats, a_stats)

        # Sort Outcomes
        probable = []
        likely = []
        unlikely = []

        for outcome, prob in results.items():
            entry = f"{outcome} ({prob:.1f}%)"
            if prob >= 75: probable.append(entry)
            elif 50 <= prob < 75: likely.append(entry)
            else: unlikely.append(entry)

        # Output formatting
        print(f"\nMATCH: {home_name} vs {away_name} ({league})")
        print(f"COEFFICIENTS: Home C_W: {cw_h} | Away C_W: {cw_a}")
        print(f"  [PROBABLE >75%]: {', '.join(probable) if probable else 'None'}")
        print(f"  [LIKELY 50-74%]: {', '.join(likely) if likely else 'None'}")
        print(f"  [UNLIKELY <50%]: {', '.join(unlikely) if unlikely else 'None'}")
        print("-" * 30)
        
        time.sleep(1) # Prevent API rate limiting

if __name__ == "__main__":
    main()
