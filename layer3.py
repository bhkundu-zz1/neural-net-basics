import numpy as np

# States: 0=Bull, 1=Bear, 2=Stagnant
# Transition matrix estimated from historical data
transition_matrix = np.array([
    [0.78, 0.14, 0.08],  # From Bull: 78% stay Bull, 14% go Bear, 8% Stagnant
    [0.20, 0.68, 0.12],  # From Bear: 20% recover, 68% stay Bear, 12% Stagnant
    [0.25, 0.23, 0.52],  # From Stagnant: 25% Bull, 23% Bear, 52% stay
])

def get_next_regime(current_state: int, steps: int = 5) -> np.ndarray:
    """Returns probability distribution over states after N steps."""
    state_vec = np.zeros(3)
    state_vec[current_state] = 1.0
    
    for _ in range(steps):
        state_vec = state_vec @ transition_matrix
    
    return {
        "bull_probability":     round(state_vec[0], 3),
        "bear_probability":     round(state_vec[1], 3),
        "stagnant_probability": round(state_vec[2], 3),
        "dominant_regime":      ["Bull", "Bear", "Stagnant"][np.argmax(state_vec)]
    }