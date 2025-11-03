import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evacuation import (
    simulate,
    room_labels as gen_room_labels,
    rank_earthquake_priorities,
    rank_fire_equipment_priorities,
)

st.set_page_config(page_title="Emergency Evacuation Planner", layout="wide")

st.title("Emergency Evacuation Planner")
st.caption("Plan evacuation priorities for Fire and Earthquake scenarios using AHP + ABC.")

with st.sidebar:
    st.header("Parameters")
    floors = st.number_input("Floors", min_value=1, max_value=20, value=4, step=1)
    rooms_per_floor = st.number_input("Rooms per Floor", min_value=1, max_value=50, value=4, step=1)

    st.subheader("AHP Importance Weights")
    # Fire: separate importance sliders
    fire_risk_importance = st.slider("Fire: Risk importance", min_value=1.0, max_value=9.0, value=4.0, step=0.5)
    fire_distance_importance = st.slider("Fire: Distance importance", min_value=1.0, max_value=9.0, value=1.0, step=0.5)
    # Earthquake: separate importance sliders
    eq_distance_importance = st.slider("Earthquake: Distance importance", min_value=1.0, max_value=9.0, value=3.0, step=0.5)
    eq_risk_importance = st.slider("Earthquake: Risk importance", min_value=1.0, max_value=9.0, value=1.0, step=0.5)

    # Convert to pairwise ratios for AHP matrices used in simulation
    fire_ratio = max(1e-6, float(fire_risk_importance) / float(fire_distance_importance))
    eq_ratio = max(1e-6, float(eq_distance_importance) / float(eq_risk_importance))

    st.subheader("Scenario Inputs")
    intensity = st.number_input("Earthquake Intensity (e.g., 4.5)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
    available_labels = gen_room_labels(int(floors), int(rooms_per_floor))
    fire_selected_rooms = st.multiselect("Fire: Rooms to prioritize equipment installation", options=available_labels, default=[])

    st.subheader("ABC Algorithm Settings")
    colony_size = st.number_input("Colony Size", min_value=5, max_value=200, value=20, step=1)
    max_iter = st.number_input("Max Iterations", min_value=10, max_value=1000, value=50, step=10)
    limit = st.number_input("Scout Limit", min_value=5, max_value=200, value=10, step=1)

    # Estimated runtime (heuristic): O(colony_size * max_iter * rooms)
    try:
        n_rooms = int(floors) * int(rooms_per_floor)
        coeff = 1.5e-4  # ~0.00015s per unit of (room * iter * colony), heuristic
        est_seconds = max(0.1, min(60.0, float(colony_size) * float(max_iter) * float(n_rooms) * coeff))
        low = est_seconds * 0.8
        high = est_seconds * 1.2

        def fmt_secs(s: float) -> str:
            return f"{s*1000:.0f} ms" if s < 1.0 else f"{s:.1f} s"

        st.caption(f"Estimated run time: ~{fmt_secs(low)}–{fmt_secs(high)}")
    except Exception:
        pass

    run = st.button("Run Simulation")

if run:
    results = simulate(
        floors=int(floors),
        rooms_per_floor=int(rooms_per_floor),
        colony_size=int(colony_size),
        max_iter=int(max_iter),
        limit=int(limit),
        fire_ratio=float(fire_ratio),
        eq_ratio=float(eq_ratio),
        seed=42,
    )

    st.success("Simulation complete.")
    # Group Earthquake and Fire views into separate tabs
    labels = results["room_labels"]
    x = np.arange(len(labels))
    fire_scores = results["fire_output"]["Evacuation_Priority"].values
    eq_scores = results["eq_output"]["Evacuation_Priority"].values

    tab_eq, tab_fire, tab_both = st.tabs(["Earthquake", "Fire", "Combined"])

    with tab_eq:
        st.metric(label="Earthquake Best Fitness", value=f"{results['eq_best_fitness']:.4f}")
        fig_eq, ax_eq = plt.subplots(figsize=(12, 4))
        ax_eq.plot(x, eq_scores, color='blue', linewidth=2)
        ax_eq.set_title("Earthquake Evacuation Priority per Room")
        ax_eq.set_xlabel("Room Number")
        ax_eq.set_ylabel("Priority Score")
        ax_eq.set_ylim(0, 0.6)
        ax_eq.set_xticks(x)
        ax_eq.set_xticklabels(labels, rotation=45, ha='right')
        ax_eq.grid(True, linestyle='--', alpha=0.7)
        fig_eq.tight_layout()
        st.pyplot(fig_eq)

        st.subheader("Earthquake: Evacuation Priority Ranking")
        eq_rank = rank_earthquake_priorities(
            results["earthquake_data"],
            results["attributes"],
            intensity=float(intensity),
        )
        st.dataframe(eq_rank, use_container_width=True)

        eq_rank_csv = eq_rank.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Earthquake Ranking (CSV)",
            data=eq_rank_csv,
            file_name="Earthquake_Evacuation_Priorities.csv",
            mime="text/csv",
            key="eq_rank_dl_tab",
        )

        st.subheader("Earthquake Scenario Data")
        st.dataframe(results["eq_output"], width='stretch')
        eq_csv = results["eq_output"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Earthquake Results (CSV)",
            data=eq_csv,
            file_name="Earthquake_Evacuation_Route.csv",
            mime="text/csv",
            key="eq_results_dl_tab",
        )

        st.subheader("Real-Life Scenario Interpretation")
        st.markdown(
            f"""
Let's imagine this is a real office building with {int(floors)} floors and {int(rooms_per_floor)} rooms on each floor. We've used our model to figure out the best way to evacuate people during an earthquake.

Looking at the separate charts and the data:

- **Earthquake Scenario (right chart):** Rooms on upper floors tend to have a higher evacuation priority due to the increased risk of structural damage. The model prioritizes getting people out of the more unstable areas first. Distance to an exit also plays a significant role here.

**In Practice:**
- Develop tailored evacuation plans: Fire and earthquake emergencies can have different priorities and routes.
- Train occupants: People in high-priority rooms for a specific scenario receive targeted training on their best evacuation routes.
- Place resources: Emergency supplies or personnel can be positioned near rooms with high evacuation priority for each scenario.
- Design future buildings: Use insights to minimize risks and optimize evacuation in different emergencies.

Considering both risk and distance, and weighting them differently based on the emergency type, provides a more nuanced and potentially safer evacuation strategy than simply evacuating based on proximity to an exit alone.
            """
        )

    with tab_fire:
        st.metric(label="Fire Best Fitness", value=f"{results['fire_best_fitness']:.4f}")
        fig_fire, ax_fire = plt.subplots(figsize=(12, 4))
        ax_fire.plot(x, fire_scores, color='red', linewidth=2)
        ax_fire.set_title("Fire Evacuation Priority per Room")
        ax_fire.set_xlabel("Room Number")
        ax_fire.set_ylabel("Priority Score")
        ax_fire.set_ylim(0, 0.6)
        ax_fire.set_xticks(x)
        ax_fire.set_xticklabels(labels, rotation=45, ha='right')
        ax_fire.grid(True, linestyle='--', alpha=0.7)
        fig_fire.tight_layout()
        st.pyplot(fig_fire)

        st.subheader("Fire: Safety Equipment Installation Ranking")
        fire_rank = rank_fire_equipment_priorities(
            results["fire_data"],
            results["attributes"],
            rooms_filter=fire_selected_rooms,
        )
        st.dataframe(fire_rank, use_container_width=True)

        fire_rank_csv = fire_rank.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Fire Equipment Ranking (CSV)",
            data=fire_rank_csv,
            file_name="Fire_Safety_Equipment_Priorities.csv",
            mime="text/csv",
            key="fire_rank_dl_tab",
        )

        st.subheader("Fire Scenario Data")
        st.dataframe(results["fire_output"], width='stretch')
        fire_csv = results["fire_output"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Fire Results (CSV)",
            data=fire_csv,
            file_name="Fire_Evacuation_Route.csv",
            mime="text/csv",
            key="fire_results_dl_tab",
        )

        st.subheader("Real-Life Scenario Interpretation")
        st.markdown(
            f"""
Let's imagine this is a real office building with {int(floors)} floors and {int(rooms_per_floor)} rooms on each floor. We've used our model to figure out the best way to evacuate people during a fire.

Looking at the separate charts and the data:

- **Fire Scenario (left chart):** Some rooms on the lower floors (where fire risk is higher) have a higher evacuation priority, especially those closest to potential fire sources or exits that might become blocked. The model prioritizes getting people out of these high-risk, potentially-blocked areas quickly.

**In Practice:**
- Develop tailored evacuation plans: Fire and earthquake emergencies can have different priorities and routes.
- Train occupants: People in high-priority rooms for a specific scenario receive targeted training on their best evacuation routes.
- Place resources: Emergency supplies or personnel can be positioned near rooms with high evacuation priority for each scenario.
- Design future buildings: Use insights to minimize risks and optimize evacuation in different emergencies.

Considering both risk and distance, and weighting them differently based on the emergency type, provides a more nuanced and potentially safer evacuation strategy than simply evacuating based on proximity to an exit alone.
            """
        )

    with tab_both:
        st.subheader("Evacuation Priority Charts")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.metric(label="Earthquake Best Fitness", value=f"{results['eq_best_fitness']:.4f}")
            fig_eq2, ax_eq2 = plt.subplots(figsize=(12, 4))
            ax_eq2.plot(x, eq_scores, color='blue', linewidth=2)
            ax_eq2.set_title("Earthquake Evacuation Priority per Room")
            ax_eq2.set_xlabel("Room Number")
            ax_eq2.set_ylabel("Priority Score")
            ax_eq2.set_ylim(0, 0.6)
            ax_eq2.set_xticks(x)
            ax_eq2.set_xticklabels(labels, rotation=45, ha='right')
            ax_eq2.grid(True, linestyle='--', alpha=0.7)
            fig_eq2.tight_layout()
            st.pyplot(fig_eq2)

        with chart_right:
            st.metric(label="Fire Best Fitness", value=f"{results['fire_best_fitness']:.4f}")
            fig_fire2, ax_fire2 = plt.subplots(figsize=(12, 4))
            ax_fire2.plot(x, fire_scores, color='red', linewidth=2)
            ax_fire2.set_title("Fire Evacuation Priority per Room")
            ax_fire2.set_xlabel("Room Number")
            ax_fire2.set_ylabel("Priority Score")
            ax_fire2.set_ylim(0, 0.6)
            ax_fire2.set_xticks(x)
            ax_fire2.set_xticklabels(labels, rotation=45, ha='right')
            ax_fire2.grid(True, linestyle='--', alpha=0.7)
            fig_fire2.tight_layout()
            st.pyplot(fig_fire2)

        st.subheader("Combined Evacuation Priority per Room")
        fig_comb, ax_comb = plt.subplots(figsize=(12, 4))
        ax_comb.plot(x, eq_scores, color='blue', linewidth=2, label='Earthquake')
        ax_comb.plot(x, fire_scores, color='red', linewidth=2, label='Fire')
        ax_comb.set_title("Combined Evacuation Priority per Room")
        ax_comb.set_xlabel("Room Number")
        ax_comb.set_ylabel("Priority Score")
        ax_comb.set_ylim(0, 0.6)
        ax_comb.set_xticks(x)
        ax_comb.set_xticklabels(labels, rotation=45, ha='right')
        ax_comb.grid(True, linestyle='--', alpha=0.7)
        ax_comb.legend(loc='upper right')
        fig_comb.tight_layout()
        st.pyplot(fig_comb)

        st.subheader("Rankings")
        rank_left, rank_right = st.columns(2)
        with rank_left:
            eq_rank_b = rank_earthquake_priorities(
                results["earthquake_data"],
                results["attributes"],
                intensity=float(intensity),
            )
            st.dataframe(eq_rank_b, use_container_width=True)
            eq_rank_b_csv = eq_rank_b.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Save Earthquake Ranking (CSV)",
                data=eq_rank_b_csv,
                file_name="Earthquake_Evacuation_Priorities.csv",
                mime="text/csv",
                key="eq_rank_dl_combined",
            )

        with rank_right:
            fire_rank_b = rank_fire_equipment_priorities(
                results["fire_data"],
                results["attributes"],
                rooms_filter=fire_selected_rooms,
            )
            st.dataframe(fire_rank_b, use_container_width=True)
            fire_rank_b_csv = fire_rank_b.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Save Fire Equipment Ranking (CSV)",
                data=fire_rank_b_csv,
                file_name="Fire_Safety_Equipment_Priorities.csv",
                mime="text/csv",
                key="fire_rank_dl_combined",
            )

        st.subheader("Real-Life Scenario Interpretation")
        st.markdown(
            f"""
Let's imagine this is a real office building with {int(floors)} floors and {int(rooms_per_floor)} rooms on each floor. We've used our model to figure out the best way to evacuate people during a fire and an earthquake.

Looking at the separate charts and the data:

- **Fire Scenario (left chart):** Some rooms on the lower floors (where fire risk is higher) have a higher evacuation priority, especially those closest to potential fire sources or exits that might become blocked. The model prioritizes getting people out of these high-risk, potentially-blocked areas quickly.

- **Earthquake Scenario (right chart):** Rooms on upper floors tend to have a higher evacuation priority due to the increased risk of structural damage. The model prioritizes getting people out of the more unstable areas first. Distance to an exit also plays a significant role here.

**In Practice:**
- Develop tailored evacuation plans: Fire and earthquake emergencies can have different priorities and routes.
- Train occupants: People in high-priority rooms for a specific scenario receive targeted training on their best evacuation routes.
- Place resources: Emergency supplies or personnel can be positioned near rooms with high evacuation priority for each scenario.
- Design future buildings: Use insights to minimize risks and optimize evacuation in different emergencies.

Considering both risk and distance, and weighting them differently based on the emergency type, provides a more nuanced and potentially safer evacuation strategy than simply evacuating based on proximity to an exit alone.
            """
        )

    # Rankings per user specification

    

    # Save CSV outputs to disk and confirm
    results["fire_output"].to_csv("Fire_Evacuation_Route.csv", index=False)
    results["eq_output"].to_csv("Earthquake_Evacuation_Route.csv", index=False)
    st.success("💾 You can save the results as Fire_Evacuation_Route.csv and Earthquake_Evacuation_Route.csv")
    st.toast("💾 You can save the results as Fire_Evacuation_Route.csv and Earthquake_Evacuation_Route.csv")
if not run:
    st.info("Set parameters in the sidebar and click 'Run Simulation'.")