import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from evac_core import demo_data, compute_priorities

st.set_page_config(page_title="Evacuation Priority Planner", layout="wide")
st.title("Evacuation Priority Planner")

st.write("Rank rooms for evacuation during **Fire** or **Earthquake** using simple, explainable scoring.")

# --- Sidebar controls --------------------------------------------------------
scenario = st.sidebar.selectbox("Scenario", ["Fire", "Earthquake"])
distance_policy = st.sidebar.selectbox(
    "Distance policy (how distance affects priority)",
    ["far_first", "near_first"],
    help="far_first: rooms farther from exits get higher priority; near_first: rooms nearer to exits get higher priority."
)

mode = st.sidebar.radio("Data source", ["Demo data", "Upload CSV"])

st.sidebar.caption(
    "Required columns: room_label, floor, distance_to_exit_m. "
    "Optional: fire_risk (for Fire) / quake_risk (for Earthquake)."
)

# --- Load data ---------------------------------------------------------------
if mode == "Demo data":
    data = demo_data()
    st.success("Using built-in demo data (20 rooms).")
else:
    up = st.file_uploader("Upload a CSV", type=["csv"])
    if not up:
        st.info(
            "Upload a CSV to continue. Expected columns: room_label, floor, distance_to_exit_m. "
            "Optional: fire_risk / quake_risk."
        )
        st.stop()
    try:
        data = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# --- Quick mapping helper (simple, required only) ----------------------------
# If column names differ, let the user map them here.
expected = ["room_label", "floor", "distance_to_exit_m"]
cols = list(data.columns)
with st.expander("Map your column names (if needed)"):
    st.caption("Match the columns in your file to what the app expects.")
    mapping = {}
    for need in expected:
        mapping[need] = st.selectbox(f"{need}", options=["(unchanged)"] + cols, index=0, key=f"map_{need}")
    # Apply mapping if selected
    rename = {src: tgt for tgt, src in mapping.items() if src != "(unchanged)"}
    if rename:
        data = data.rename(columns=rename)

# Validate required columns
missing = [c for c in expected if c not in data.columns]
if missing:
    st.error(f"Missing required columns after mapping: {missing}")
    st.stop()

# Show preview
st.subheader("Input preview")
with st.expander("Review and optionally edit data", expanded=True):
    st.caption("Edits here immediately feed into the priority computation.")
    data = st.data_editor(
        data,
        use_container_width=True,
        key="data_editor",
        num_rows="dynamic"
    )

# --- Compute priorities ------------------------------------------------------
try:
    result, meta = compute_priorities(data, scenario=scenario, distance_mode=distance_policy)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Results")
st.caption(
    f"Scenario: **{scenario}** | Weights → Risk: {meta['weights']['risk']:.3f}, "
    f"Distance: {meta['weights']['distance']:.3f} | Distance policy: **{meta['distance_mode']}**"
)
st.dataframe(
    result[["room_label", "floor", "distance_to_exit_m", meta["risk_col_used"], "priority"]].head(50),
    use_container_width=True,
)

# Plot
fig, ax = plt.subplots()
ax.plot(result["priority"].values, marker="o")
ax.set_title("Evacuation priority (higher = earlier)")
ax.set_xlabel("Ranked rooms (left = highest)")
ax.set_ylabel("Priority score")
st.pyplot(fig, use_container_width=True)

# Download CSV
buf = io.StringIO()
result.to_csv(buf, index=False)
st.download_button(
    "Download priorities CSV",
    data=buf.getvalue().encode("utf-8"),
    file_name="evacuation_priorities.csv",
    mime="text/csv",
)