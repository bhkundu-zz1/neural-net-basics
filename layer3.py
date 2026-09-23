import json
import os

import numpy as np

# States: 0=Bull, 1=Bear, 2=Stagnant

# v1 fallback: a hand-typed constant, never fit from data — kept only as the
# default for a fresh checkout where fit_layer3_transition_matrix.py hasn't
# been run yet. See docs/pipeline_guide.md's "layer3's regime label carries
# no signal" finding for why this was replaced: combined with the old
# Hurst-based seeding rule, it produced dominant_regime="Bull" 100% of the
# time, out-of-sample, with zero discriminative signal.
_FALLBACK_TRANSITION_MATRIX = np.array([
    [0.78, 0.14, 0.08],  # From Bull: 78% stay Bull, 14% go Bear, 8% Stagnant
    [0.20, 0.68, 0.12],  # From Bear: 20% recover, 68% stay Bear, 12% Stagnant
    [0.25, 0.23, 0.52],  # From Stagnant: 25% Bull, 23% Bear, 52% stay
])

_TRANSITION_MATRIX_PATH = "layer3_transition_matrix.json"
_STATE_NAMES = ["Bull", "Bear", "Stagnant"]


def _load_transition_matrix() -> np.ndarray:
    """
    Loads the empirically fitted matrix from fit_layer3_transition_matrix.py's
    output if present; falls back to the hand-typed constant otherwise (e.g.
    a fresh checkout, or a deliberately reverted/deleted fitted file).
    """
    if os.path.exists(_TRANSITION_MATRIX_PATH):
        with open(_TRANSITION_MATRIX_PATH) as f:
            data = json.load(f)
        matrix = np.array(data["transition_matrix"])
        if matrix.shape == (3, 3):
            return matrix
    return _FALLBACK_TRANSITION_MATRIX


transition_matrix = _load_transition_matrix()


def seed_state_from_prices(prices, horizon_days: int = 5, up_threshold: float = 0.02,
                            down_threshold: float = 0.02) -> int:
    """
    Seeds current_state from REALIZED price direction over the trailing
    horizon_days — replaces the old current_state = 0 if hurst "trending"
    else 2 rule, which conflated trend PERSISTENCE (Hurst exponent) with
    price DIRECTION (Bull/Bear) and could never seed Bear at all. Uses the
    same threshold definition fit_layer3_transition_matrix.py fits the
    matrix against and calibrate_layer3_regime.py validates it with, so
    seeding, fitting, and validation all agree on what "Bull"/"Bear"/
    "Stagnant" means.

    Falls back to Stagnant (state 2) if fewer than horizon_days+1 prices are
    available — matches the old code's conservative default for short
    histories.
    """
    if len(prices) < horizon_days + 1:
        return 2

    trailing_return = float(prices.iloc[-1] / prices.iloc[-(horizon_days + 1)] - 1.0)
    if trailing_return > up_threshold:
        return 0
    if trailing_return < -down_threshold:
        return 1
    return 2

# Empirical, out-of-sample check of dominant_regime against realized forward
# returns — see calibrate_layer3_regime.py and docs/pipeline_guide.md's "The
# limits" section. Mirrors trade_glue.py's calibration_table.json pattern:
# loaded once and cached. NOT attached to get_next_regime's own return dict —
# layer4.py's build_feature_vector consumes that dict's values() wholesale
# (filtering only by numeric type) to build the model's feature vector, and
# every trained checkpoint expects exactly the 3 original probability floats.
# Adding keys here would silently change the feature vector's shape/semantics
# for live inference and any future retraining. Callers that want the
# empirical context (e.g. pipeline_core.py, for display only) should call
# attach_regime_calibration() on their OWN copy, after build_feature_vector
# has already consumed the original dict.
_REGIME_CALIBRATION_TABLE_PATH = "regime_calibration_table.json"
_regime_calibration_table_cache = None


def _load_regime_calibration_table():
    global _regime_calibration_table_cache
    if _regime_calibration_table_cache is not None:
        return _regime_calibration_table_cache
    if os.path.exists(_REGIME_CALIBRATION_TABLE_PATH):
        with open(_REGIME_CALIBRATION_TABLE_PATH) as f:
            _regime_calibration_table_cache = json.load(f)
    else:
        _regime_calibration_table_cache = []
    return _regime_calibration_table_cache


def _lookup_regime_calibration(dominant_regime: str, table: list[dict]) -> dict | None:
    for entry in table:
        if entry["predicted_regime"] == dominant_regime:
            return entry
    return None


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


def attach_regime_calibration(regime_probs: dict) -> dict:
    """
    Returns a NEW dict (regime_probs plus regime_calibrated/regime_hit_rate),
    for display/API purposes only — never pass this enriched dict into
    build_feature_vector (see module docstring above).
    """
    table = _load_regime_calibration_table()
    bucket = _lookup_regime_calibration(regime_probs["dominant_regime"], table)
    return {
        **regime_probs,
        "regime_calibrated": bucket is not None,
        "regime_hit_rate": bucket["hit_rate"] if bucket is not None else None,
    }