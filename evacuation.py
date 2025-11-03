import numpy as np
import pandas as pd


def build_base_data(floors: int = 4, rooms_per_floor: int = 5, seed: int | None = 42) -> pd.DataFrame:
    """Create base building data with Floor, Room, and Distance to exit."""
    if seed is not None:
        np.random.seed(seed)
    total_rooms = floors * rooms_per_floor
    distances = np.random.uniform(5, 50, total_rooms)
    df = pd.DataFrame({
        "Floor": np.repeat(np.arange(1, floors + 1), rooms_per_floor),
        "Room": np.arange(1, total_rooms + 1),
        "Distance": distances,
    })
    return df


def add_building_attributes(base: pd.DataFrame,
                            floors: int,
                            rooms_per_floor: int,
                            seed: int | None = 42) -> pd.DataFrame:
    """Augment base data with occupancy, structural safety, and room purpose.

    - Occupancy: simulated occupants per room (higher occupancy increases priority).
    - StructuralSafety: simulated 0-1 safety score (lower increases priority).
    - Purpose: categorical room purpose influencing fire equipment priority.
    """
    if seed is not None:
        np.random.seed(seed + 10)
    df = base.copy()
    total_rooms = len(df)

    # Simulate occupancy (people per room), biased slightly by floor
    floor_bias = 1 + (df["Floor"] / max(df["Floor"])) * 0.2
    occupancy = (np.random.randint(5, 50, size=total_rooms) * floor_bias).astype(int)

    # Structural safety decreases slightly with higher floors; add small noise
    safety_base = 0.9 - (df["Floor"] - 1) / max(df["Floor"]) * 0.3
    structural_safety = np.clip(safety_base + np.random.uniform(-0.05, 0.05, total_rooms), 0.3, 1.0)

    # Room purposes
    purposes = ["Office", "Meeting", "Lab", "Storage", "Kitchen", "Server"]
    purpose_probs = [0.35, 0.2, 0.1, 0.15, 0.1, 0.1]
    purpose = np.random.choice(purposes, size=total_rooms, p=purpose_probs)

    df["Occupancy"] = occupancy
    df["StructuralSafety"] = structural_safety
    df["Purpose"] = purpose
    return df


def fire_scenario_data(base: pd.DataFrame, seed: int | None = 42) -> pd.DataFrame:
    """Add fire risk to base data: higher risk on lower floors and near exits."""
    if seed is not None:
        np.random.seed(seed + 1)
    total_rooms = len(base)
    fire_risk = np.linspace(1.0, 0.5, total_rooms) + np.random.uniform(-0.1, 0.1, total_rooms)
    fire_risk = np.clip(fire_risk, 0.3, 1.0)
    df = base.copy()
    df["Risk"] = fire_risk
    return df


def earthquake_scenario_data(base: pd.DataFrame, floors: int, rooms_per_floor: int, seed: int | None = 42) -> pd.DataFrame:
    """Add earthquake risk: higher risk on upper floors due to instability."""
    if seed is not None:
        np.random.seed(seed + 2)
    total_rooms = len(base)
    earthquake_risk = np.repeat(np.linspace(0.5, 1.0, floors), rooms_per_floor)
    earthquake_risk += np.random.uniform(-0.05, 0.05, total_rooms)
    earthquake_risk = np.clip(earthquake_risk, 0.3, 1.0)
    df = base.copy()
    df["Risk"] = earthquake_risk
    return df


def ahp_weights(criteria_matrix: np.ndarray) -> np.ndarray:
    """Compute AHP weights via principal eigenvector."""
    eigvals, eigvecs = np.linalg.eig(criteria_matrix)
    max_index = np.argmax(eigvals)
    weights = np.real(eigvecs[:, max_index])
    weights = weights / np.sum(weights)
    return np.real(weights)


def fitness_function(solution_scaled: np.ndarray, weights: np.ndarray) -> float:
    """Lower risk and shorter distance => higher fitness.
    solution_scaled: array of shape (n_rooms, 2) representing [risk, distance] after scaling.
    weights: length-2 array for [risk_weight, distance_weight].
    """
    risk_score = np.sum(solution_scaled[:, 0] * weights[0])
    distance_score = np.sum(solution_scaled[:, 1] * weights[1])
    return 1.0 / (risk_score + distance_score)


def artificial_bee_colony(data_two_cols: np.ndarray,
                          weights: np.ndarray,
                          colony_size: int = 20,
                          max_iter: int = 50,
                          limit: int = 10,
                          seed: int | None = 42):
    """Artificial Bee Colony optimization for evacuation prioritization.

    data_two_cols: numpy array of shape (n_rooms, 2) with columns [Risk, Distance].
    weights: length-2 weights for [Risk, Distance].
    Returns: (best_solution, best_fitness)
    """
    if seed is not None:
        np.random.seed(seed)

    num_rooms = data_two_cols.shape[0]
    solutions = np.random.rand(colony_size, num_rooms, 2)
    fitness = np.array([fitness_function(data_two_cols * s, weights) for s in solutions])
    trial = np.zeros(colony_size)

    best_idx = int(np.argmax(fitness))
    best_solution = solutions[best_idx]
    best_fitness = float(fitness[best_idx])

    for _ in range(max_iter):
        # Employed bee phase
        for i in range(colony_size):
            k = np.random.randint(0, colony_size)
            while k == i:
                k = np.random.randint(0, colony_size)
            phi = np.random.uniform(-1, 1, size=solutions[i].shape)
            new_solution = solutions[i] + phi * (solutions[i] - solutions[k])
            new_solution = np.clip(new_solution, 0, 1)

            new_fitness = fitness_function(data_two_cols * new_solution, weights)
            if new_fitness > fitness[i]:
                solutions[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker bee phase
        prob = fitness / np.sum(fitness)
        for i in range(colony_size):
            if np.random.rand() < prob[i]:
                k = np.random.randint(0, colony_size)
                phi = np.random.uniform(-1, 1, size=solutions[i].shape)
                new_solution = solutions[i] + phi * (solutions[i] - solutions[k])
                new_solution = np.clip(new_solution, 0, 1)

                new_fitness = fitness_function(data_two_cols * new_solution, weights)
                if new_fitness > fitness[i]:
                    solutions[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

        # Scout bee phase
        for i in range(colony_size):
            if trial[i] > limit:
                solutions[i] = np.random.rand(num_rooms, 2)
                fitness[i] = fitness_function(data_two_cols * solutions[i], weights)
                trial[i] = 0

        # Update best
        current_best = np.max(fitness)
        if current_best > best_fitness:
            best_fitness = float(current_best)
            best_solution = solutions[int(np.argmax(fitness))]

    return best_solution, best_fitness


def room_labels(floors: int, rooms_per_floor: int) -> list[str]:
    labels = []
    for f in range(1, floors + 1):
        for r in range(1, rooms_per_floor + 1):
            labels.append(f"Room {f}{r:02d}")
    return labels


def rank_earthquake_priorities(eq_data: pd.DataFrame,
                               attributes: pd.DataFrame,
                               intensity: float,
                               weights: dict | None = None) -> pd.DataFrame:
    """Rank rooms for immediate evacuation in an earthquake.

    Criteria: higher occupancy, lower structural safety, proximity to exits (shorter distance).
    Intensity (e.g., 4.5, 6.0, 7.8) scales the impact of structural safety.
    """
    if weights is None:
        weights = {"occupancy": 0.4, "safety": 0.4, "exit_proximity": 0.2}

    df = eq_data.merge(attributes[["Room", "Occupancy", "StructuralSafety"]], on="Room", how="left")

    # Normalize features
    occ_norm = (df["Occupancy"] - df["Occupancy"].min()) / (df["Occupancy"].max() - df["Occupancy"].min() + 1e-9)
    safety_norm = (df["StructuralSafety"] - df["StructuralSafety"].min()) / (df["StructuralSafety"].max() - df["StructuralSafety"].min() + 1e-9)
    # Proximity to exits: closer distance => higher proximity
    dist_norm = (df["Distance"] - df["Distance"].min()) / (df["Distance"].max() - df["Distance"].min() + 1e-9)
    exit_proximity = 1.0 - dist_norm

    intensity_factor = max(0.0, min(float(intensity), 10.0)) / 10.0

    priority = (
        weights["occupancy"] * occ_norm
        + weights["safety"] * (1.0 - safety_norm) * (0.5 + 0.5 * intensity_factor)
        + weights["exit_proximity"] * exit_proximity
    )

    out = df.copy()
    out["Priority_Score"] = priority
    out = out.sort_values(by="Priority_Score", ascending=False).reset_index(drop=True)
    # Include Risk and Distance so users can see contributing factors
    return out[["Floor", "Room", "Risk", "Distance", "Occupancy", "StructuralSafety", "Priority_Score"]]


def rank_fire_equipment_priorities(fire_data: pd.DataFrame,
                                   attributes: pd.DataFrame,
                                   rooms_filter: list[str] | list[int] | None = None,
                                   weights: dict | None = None) -> pd.DataFrame:
    """Rank rooms to equip with fire safety tools based on risk, room purpose, and accessibility.

    - Risk: from fire scenario data (higher => higher priority)
    - Purpose: category weight (e.g., Lab > Server > Kitchen > Office > Meeting > Storage)
    - Accessibility: proximity to exits (closer => easier to equip quickly)
    rooms_filter can be labels like "Room 101" or integer room ids; if None, include all.
    """
    if weights is None:
        weights = {"risk": 0.5, "purpose": 0.3, "access": 0.2}

    df = fire_data.merge(attributes[["Room", "Purpose"]], on="Room", how="left")

    # Map purpose to weights
    purpose_weight_map = {
        "Lab": 1.0,
        "Server": 0.9,
        "Kitchen": 0.8,
        "Office": 0.6,
        "Meeting": 0.5,
        "Storage": 0.4,
    }
    purpose_weight = df["Purpose"].map(purpose_weight_map).fillna(0.5)

    # Accessibility via proximity to exits
    dist_norm = (df["Distance"] - df["Distance"].min()) / (df["Distance"].max() - df["Distance"].min() + 1e-9)
    access = 1.0 - dist_norm

    risk_norm = (df["Risk"] - df["Risk"].min()) / (df["Risk"].max() - df["Risk"].min() + 1e-9)

    priority = (
        weights["risk"] * risk_norm
        + weights["purpose"] * purpose_weight
        + weights["access"] * access
    )

    out = df.copy()
    out["Equipment_Priority_Score"] = priority

    # Optional filtering
    if rooms_filter:
        # Derive rooms_per_floor from data to map labels like "Room 101"
        try:
            # Prefer the most common count of rooms per floor to handle irregularities
            rooms_per_floor = int(fire_data.groupby("Floor")["Room"].count().mode().iloc[0])
        except Exception:
            # Fallback: estimate from first floor
            rooms_per_floor = int(fire_data[fire_data["Floor"] == fire_data["Floor"].min()].shape[0])

        # Normalize filter to room ids
        norm_filter_ids: list[int] = []
        for r in rooms_filter:
            if isinstance(r, int):
                norm_filter_ids.append(r)
            elif isinstance(r, str):
                # Expect labels like "Room 101" -> parse 3-digit number or fallback
                parts = r.strip().split()
                if len(parts) == 2 and parts[0].lower() == "room":
                    try:
                        label_num = int(parts[1])
                        # Map label like 101 to room index: floor first digit, room two digits
                        floor = label_num // 100
                        room_no = label_num % 100
                        # Convert to sequential id
                        rid = (floor - 1) * rooms_per_floor + room_no
                        norm_filter_ids.append(rid)
                    except Exception:
                        pass
        if norm_filter_ids:
            out = out[out["Room"].isin(norm_filter_ids)]

    out = out.sort_values(by="Equipment_Priority_Score", ascending=False).reset_index(drop=True)
    # Include Risk and Distance for transparency
    return out[["Floor", "Room", "Risk", "Distance", "Purpose", "Equipment_Priority_Score"]]


def simulate(floors: int = 4,
             rooms_per_floor: int = 5,
             colony_size: int = 20,
             max_iter: int = 50,
             limit: int = 10,
             fire_ratio: float = 4.0,
             eq_ratio: float = 3.0,
             seed: int | None = 42):
    """Run both fire and earthquake scenarios and return results.

    fire_ratio: risk:distance importance ratio (>1 means risk more important).
    eq_ratio: distance:risk importance ratio (>1 means distance more important).
    """
    base = build_base_data(floors=floors, rooms_per_floor=rooms_per_floor, seed=seed)
    attributes = add_building_attributes(base, floors=floors, rooms_per_floor=rooms_per_floor, seed=seed)

    fire_data = fire_scenario_data(base, seed=seed)
    earthquake_data = earthquake_scenario_data(base, floors=floors, rooms_per_floor=rooms_per_floor, seed=seed)

    # AHP weights
    fire_matrix = np.array([[1.0, fire_ratio], [1.0 / fire_ratio, 1.0]])
    fire_weights = ahp_weights(fire_matrix)

    eq_matrix = np.array([[1.0, 1.0 / eq_ratio], [eq_ratio, 1.0]])
    eq_weights = ahp_weights(eq_matrix)

    # Optimize
    fire_best_solution, fire_best_fitness = artificial_bee_colony(
        fire_data[["Risk", "Distance"]].values,
        fire_weights,
        colony_size=colony_size,
        max_iter=max_iter,
        limit=limit,
        seed=seed,
    )

    eq_best_solution, eq_best_fitness = artificial_bee_colony(
        earthquake_data[["Risk", "Distance"]].values,
        eq_weights,
        colony_size=colony_size,
        max_iter=max_iter,
        limit=limit,
        seed=seed,
    )

    # Outputs
    fire_output = fire_data.copy()
    fire_output["Evacuation_Priority"] = np.mean(fire_best_solution, axis=1)

    eq_output = earthquake_data.copy()
    eq_output["Evacuation_Priority"] = np.mean(eq_best_solution, axis=1)

    return {
        "base": base,
        "attributes": attributes,
        "fire_data": fire_data,
        "earthquake_data": earthquake_data,
        "fire_best_solution": fire_best_solution,
        "fire_best_fitness": fire_best_fitness,
        "eq_best_solution": eq_best_solution,
        "eq_best_fitness": eq_best_fitness,
        "fire_output": fire_output,
        "eq_output": eq_output,
        "room_labels": room_labels(floors, rooms_per_floor),
    }