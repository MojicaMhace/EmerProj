import numpy as np
import pandas as pd

# ---- Utilities --------------------------------------------------------------

def ahp_weights(matrix):
    """Return AHP weights from a 2x2 (or nxn) pairwise comparison matrix."""
    m = np.array(matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eig(m)
    idx = np.argmax(eigvals.real)
    w = eigvecs[:, idx].real
    w = w / w.sum()
    return w

def _normalize(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    rng = (hi - lo) if hi > lo else 1.0
    return (s - lo) / rng

# ---- Demo data --------------------------------------------------------------

def demo_data(floors=4, rooms_per_floor=5, seed=42) -> pd.DataFrame:
    """Create a small synthetic building (4 floors x 5 rooms = 20 rooms)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(1, floors + 1):
        for r in range(1, rooms_per_floor + 1):
            room = f * 100 + r
            dist = int(rng.integers(8, 51))  # meters
            fire_base = 0.9 - 0.15 * (f - 1)   # lower floors riskier in fire
            quake_base = 0.3 + 0.20 * (f - 1)  # higher floors riskier in quake
            fire_risk = float(np.clip(fire_base + rng.normal(0, 0.05), 0.05, 0.95))
            quake_risk = float(np.clip(quake_base + rng.normal(0, 0.05), 0.05, 0.95))
            rows.append({
                "room_label": f"Room {room}",
                "floor": f,
                "distance_to_exit_m": dist,
                "fire_risk": round(fire_risk, 3),
                "quake_risk": round(quake_risk, 3),
            })
    return pd.DataFrame(rows)

# ---- Core scoring -----------------------------------------------------------

def compute_priorities(df: pd.DataFrame,
                       scenario: str = "Fire",
                       distance_mode: str = "far_first"):
    """
    Compute a simple, explainable priority per room.

    distance_mode:
      - "far_first"   -> rooms farther from the exit get higher priority
      - "near_first"  -> rooms closer to the exit get higher priority
    """
    scenario = scenario.lower().strip()
    if scenario not in {"fire", "earthquake"}:
        raise ValueError("scenario must be 'Fire' or 'Earthquake'")

    needed = ["room_label", "floor", "distance_to_exit_m"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    # Pick/derive a risk column for this scenario
    risk_col = "fire_risk" if scenario == "fire" else "quake_risk"
    if risk_col not in df.columns:
        # Derive a simple risk from floor if user didn't supply one.
        # Fire: lower floors riskier; Quake: higher floors riskier.
        f = df["floor"].astype(float)
        risk_proxy = (f.max() - f + 1) if scenario == "fire" else f
        df[risk_col] = _normalize(risk_proxy)

    # Normalize risk & distance
    df["_risk_norm"] = _normalize(df[risk_col])
    dist_norm = _normalize(df["distance_to_exit_m"])
    if distance_mode == "near_first":
        df["_dist_component"] = 1.0 - dist_norm
    else:
        df["_dist_component"] = dist_norm

    # AHP weights for 2 criteria: [Risk, Distance]
    if scenario == "fire":
        # Risk >> Distance (example: 4x more important)
        w_risk, w_dist = ahp_weights([[1, 4],
                                      [1/4, 1]])
    else:
        # Distance >> Risk (example: 3x more important)
        w_risk, w_dist = ahp_weights([[1, 1/3],
                                      [3, 1]])

    # Simple, transparent formula:
    # higher priority = higher (weighted) risk + chosen distance component
    df["priority"] = w_risk * df["_risk_norm"] + w_dist * df["_dist_component"]

    # Sort: highest priority first
    df = df.sort_values("priority", ascending=False).reset_index(drop=True)

    meta = {
        "scenario": scenario,
        "risk_col_used": risk_col,
        "weights": {"risk": float(w_risk), "distance": float(w_dist)},
        "distance_mode": distance_mode
    }

    # Clean up helper cols in the returned table
    return df.drop(columns=["_risk_norm", "_dist_component"]), meta