import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
import json

from evacuation import (
    simulate,
    room_labels as gen_room_labels,
    rank_earthquake_priorities,
    rank_fire_equipment_priorities,
)
from evac_core import demo_data as core_demo_data, compute_priorities as core_compute_priorities

def build_fire_equipment_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    top10 = max(1, int(np.ceil(0.1 * n)))
    top30 = max(1, int(np.ceil(0.3 * n)))
    base_map = {
        "Lab": ["CO2 extinguisher", "ABC extinguisher", "Fire blanket", "Smoke detector"],
        "Server": ["CO2 extinguisher", "Clean agent extinguisher", "Smoke detector"],
        "Canteen": ["Class K extinguisher", "Fire blanket", "Smoke detector"],
        "Office": ["ABC extinguisher", "Smoke detector"],
        "Faculty": ["ABC extinguisher", "Smoke detector"],
        "Storage": ["ABC extinguisher", "Smoke detector"],
    }
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        rank = i + 1
        purpose = str(row["Purpose"]) if "Purpose" in df.columns else "Office"
        base = base_map.get(purpose, ["ABC extinguisher", "Smoke detector"])
        extras = []
        if rank <= top10:
            extras.extend(["Additional ABC extinguisher", "Emergency light"])
        elif rank <= top30:
            extras.append("Emergency light")
        equip = []
        for item in base + extras:
            if item not in equip:
                equip.append(item)
        rows.append({
            "Floor": row["Floor"],
            "Room": row["Room"],
            "Purpose": purpose,
            "Rank": rank,
            "Recommended_Equipment": ", ".join(equip)
        })
    return pd.DataFrame(rows)

EQUIPMENT_ASSET_DIR = os.path.join("assets", "equipment")
_EQUIPMENT_FILES = {
    "CO2 extinguisher": "co2_extinguisher.png",
    "Clean agent extinguisher": "clean_agent_extinguisher.png",
    "ABC extinguisher": "abc_extinguisher.png",
    "Additional ABC extinguisher": "abc_extinguisher.png",
    "Smoke detector": "smoke_detector.png",
    "Fire blanket": "fire_blanket.png",
    "Emergency light": "emergency_light.png",
    "Class K extinguisher": "class_k_extinguisher.png",
    "Hard hat": "hard_hat.png",
    "First aid kit": "first_aid_kit.png",
    "Whistle": "whistle.png",
    "Emergency radio": "emergency_radio.png",
    "Emergency blanket": "emergency_blanket.png",
}
_PLACEHOLDER_BASE = "https://via.placeholder.com/320x200.png?text="
_UNKNOWN_EQUIPMENT_IMAGE = f"{_PLACEHOLDER_BASE}Equipment"


def _build_equipment_image_map() -> dict:
    image_map = {}
    for name, filename in _EQUIPMENT_FILES.items():
        local_path = os.path.join(EQUIPMENT_ASSET_DIR, filename)
        if os.path.exists(local_path):
            image_map[name] = local_path
        else:
            image_map[name] = f"{_PLACEHOLDER_BASE}{name.replace(' ', '+')}"
    image_map["unknown"] = _UNKNOWN_EQUIPMENT_IMAGE
    return image_map


equipment_images = _build_equipment_image_map()

def _build_equipment_web_src_map() -> dict:
    web_map = {}
    for name, src in equipment_images.items():
        if isinstance(src, str) and os.path.exists(src):
            try:
                with open(src, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                web_map[name] = f"data:image/png;base64,{b64}"
            except Exception:
                web_map[name] = _UNKNOWN_EQUIPMENT_IMAGE
        else:
            web_map[name] = src
    web_map["unknown"] = web_map.get("unknown", _UNKNOWN_EQUIPMENT_IMAGE)
    return web_map

def _parse_equipment_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [x.strip() for x in str(value).split(",")]
    return [x for x in parts if x]

def render_equipment_gallery():
    valid_items = {k: v for k, v in equipment_images.items() if k != "unknown"}
    names = list(valid_items.keys())
    n_cols = 4
    cols = st.columns(n_cols)
    for i, name in enumerate(names):
        with cols[i % n_cols]:
            st.image(valid_items[name], caption=name, use_container_width=True)

@st.dialog("Equipment Gallery")
def show_gallery_modal():
    render_equipment_gallery()

def render_gallery_toggle_button(key_prefix):
    if st.button("Show Equipment Images", key=f"btn_{key_prefix}"):
        show_gallery_modal()

_EQUIP_TABLE_CSS = """
<style>
:root {
    --text-color: #ffffff;
    --secondary-background-color: #262730;
    --primary-color: #4ea1ff;
    --background-color: #0e1117;
}
body { margin: 0; padding: 0; }
.equip-wrap {
    width: 100%;
    border: 1px solid rgba(250,250,250,0.2);
    border-radius: 0.25rem;
    overflow: hidden;
    font-family: "Source Sans Pro", sans-serif;
    margin-bottom: 0px;
}
.equip-scroll {
    max-height: 400px;
    overflow: auto;
}
.equip-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    color: var(--text-color);
    background-color: transparent;
    border: 1px solid rgba(250,250,250,0.2);
}
.equip-table th, .equip-table td {
    padding: 8px 12px;
    border: 1px solid rgba(250,250,250,0.2);
    color: var(--text-color);
    vertical-align: middle;
    white-space: nowrap !important;
    text-align: left;
}
.equip-table th {
    position: sticky;
    top: 0;
    background-color: var(--secondary-background-color);
    font-weight: 600;
    border: 1px solid rgba(250,250,250,0.2);
    z-index: 2;
}
.equip-table tr:hover { background-color: rgba(255,255,255,0.05); }
.equip-item {
    cursor: pointer;
    color: var(--primary-color);
    text-decoration: none;
}
.equip-item:hover { text-decoration: underline; }
.equip-modal {
    display: none;
    position: fixed;
    z-index: 9999;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.6);
}
.equip-modal-content {
    background-color: #1e1e1e;
    margin: 10% auto;
    padding: 20px;
    border: 1px solid #444;
    width: 80%;
    max-width: 700px;
    border-radius: 8px;
    position: relative;
    color: var(--text-color);
}
.equip-close {
    color: #aaa;
    float: right;
    font-size: 28px;
    font-weight: 600;
    cursor: pointer;
}
.equip-close:hover { color: #fff; }
.equip-caption {
    margin-top: 10px;
    font-weight: 600;
    color: var(--text-color);
    text-align: center;
}
</style>
"""

def render_clickable_equipment_table(equip_df: pd.DataFrame, key_prefix: str):
    cols = list(equip_df.columns)
    web_src = _build_equipment_web_src_map()
    modal_id = f"equipModal_{key_prefix}"
    img_id = f"{modal_id}_img"
    caption_id = f"{modal_id}_caption"
    html = [_EQUIP_TABLE_CSS]
    html.append(f"<div id='{modal_id}' class='equip-modal'><div class='equip-modal-content'>")
    html.append(f"<span class='equip-close' id='{modal_id}_close'>&times;</span>")
    html.append(f"<img id='{img_id}' src='' style='max-width:100%;height:auto'/>")
    html.append(f"<div id='{caption_id}' class='equip-caption'></div>")
    html.append("</div></div>")
    html.append("<div class='equip-wrap'><div class='equip-scroll'>")
    html.append("<table class='equip-table'>")
    html.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead>")
    html.append("<tbody>")
    for _, row in equip_df.iterrows():
        html.append("<tr>")
        for c in cols:
            if c == "Recommended_Equipment":
                items = _parse_equipment_list(row[c])
                content = ", ".join([f"<a class='equip-item' data-equip=\"{x}\">{x}</a>" for x in items]) or ""
                html.append(f"<td>{content}</td>")
            else:
                html.append(f"<td>{row[c]}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div></div>")
    img_map_json = json.dumps(web_src)
    script = f"""
    <script>
    const IMG_MAP = {img_map_json};
    const modal = document.getElementById("{modal_id}");
    const imgEl = document.getElementById("{img_id}");
    const captionEl = document.getElementById("{caption_id}");
    const closeBtn = document.getElementById("{modal_id}_close");
    function showEquip(name) {{
        let src = IMG_MAP[name] || IMG_MAP[(name || '').replace("Additional ", "")] || IMG_MAP["unknown"];
        imgEl.src = src;
        captionEl.textContent = name;
        modal.style.display = "block";
    }}
    document.querySelectorAll("a.equip-item").forEach(a => {{
        a.addEventListener("click", (e) => {{
            e.preventDefault();
            const name = a.getAttribute("data-equip");
            showEquip(name);
        }});
    }});
    closeBtn.onclick = () => {{ modal.style.display = "none"; }};
    window.addEventListener("click", (e) => {{
        if (e.target === modal) modal.style.display = "none";
    }});
    </script>
    """
    html.append(script)
    
    # Dynamic height calculation
    row_height = 35  # Approximate height per row in pixels
    header_height = 40 # Approximate header height
    max_height = 410 # Matches CSS max-height + small buffer
    
    # Calculate required height based on number of rows
    # Add a small buffer for borders/margins
    calculated_height = header_height + (len(equip_df) * row_height) + 15
    
    # Clamp to max_height, but ensure at least some minimum
    final_height = min(calculated_height, max_height)
    
    components.html("".join(html), height=final_height, scrolling=False)

def build_eq_equipment_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    top10 = max(1, int(np.ceil(0.1 * n)))
    top30 = max(1, int(np.ceil(0.3 * n)))
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        rank = i + 1
        items = []
        if rank <= top10:
            items = ["Hard hat", "First aid kit", "Emergency light", "Whistle", "Emergency radio", "Emergency blanket"]
        elif rank <= top30:
            items = ["First aid kit", "Emergency light", "Whistle"]
        else:
            items = ["Emergency light"]
        rows.append({
            "Floor": row["Floor"],
            "Room": row["Room"],
            "Rank": rank,
            "Recommended_Equipment": ", ".join(items)
        })
    return pd.DataFrame(rows)

st.set_page_config(page_title="Emergency Evacuation Planner", layout="wide")

st.title("Emergency Evacuation Planner")
st.caption("Plan evacuation priorities for Fire and Earthquake scenarios using AHP + ABC.")

with st.sidebar:
    st.header("Parameters")
    data_mode = st.radio("Data source", options=["Demo data", "Upload CSV"], index=0)
    floors = st.number_input("Floors", min_value=1, max_value=20, value=4, step=1)
    rooms_per_floor = st.number_input("Rooms per Floor", min_value=1, max_value=50, value=4, step=1)

    available_labels = gen_room_labels(int(floors), int(rooms_per_floor))
    with st.expander("Fire Settings", expanded=True):
        fire_risk_importance = st.slider("Risk importance", min_value=1.0, max_value=9.0, value=4.0, step=0.5)
        fire_distance_importance = st.slider("Distance importance", min_value=1.0, max_value=9.0, value=1.0, step=0.5)
        fire_selected_rooms = st.multiselect("Rooms to prioritize equipment installation", options=available_labels, default=[], key="fire_rooms_select")
    with st.expander("Earthquake Settings", expanded=True):
        eq_distance_importance = st.slider("Distance importance", min_value=1.0, max_value=9.0, value=3.0, step=0.5)
        eq_risk_importance = st.slider("Risk importance", min_value=1.0, max_value=9.0, value=1.0, step=0.5)
        intensity = st.number_input("Intensity (e.g., 4.5)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
        eq_selected_rooms = st.multiselect("Rooms to prioritize evacuation", options=available_labels, default=[], key="eq_rooms_select")

    fire_ratio = max(1e-6, float(fire_risk_importance) / float(fire_distance_importance))
    eq_ratio = max(1e-6, float(eq_distance_importance) / float(eq_risk_importance))

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

if data_mode == "Demo data" and (run or "demo_results" in st.session_state):
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
        st.session_state["demo_results"] = results
    else:
        results = st.session_state["demo_results"]

    st.success("Simulation complete.")

    # Group Earthquake and Fire views into separate tabs
    labels = results["room_labels"]
    x = np.arange(len(labels))
    fire_scores = results["fire_output"]["Evacuation_Priority"].values
    eq_scores = results["eq_output"]["Evacuation_Priority"].values

    eq_rank = rank_earthquake_priorities(
        results["earthquake_data"],
        results["attributes"],
        intensity=float(intensity),
    ).copy()
    eq_rank.insert(0, "Rank", np.arange(1, len(eq_rank) + 1))
    if eq_selected_rooms:
        try:
            rpf = int(results["earthquake_data"].groupby("Floor")["Room"].count().mode().iloc[0])
        except Exception:
            rpf = int(rooms_per_floor)
        norm_ids = []
        for r in eq_selected_rooms:
            parts = str(r).split()
            if len(parts) == 2 and parts[0].lower() == "room":
                try:
                    num = int(parts[1])
                    floor = num // 100
                    room_no = num % 100
                    rid = (floor - 1) * rpf + room_no
                    norm_ids.append(rid)
                except Exception:
                    pass
        if norm_ids:
            eq_rank = eq_rank[eq_rank["Room"].isin(norm_ids)].reset_index(drop=True)
            eq_rank["Rank"] = np.arange(1, len(eq_rank) + 1)
    eq_rank_simple = eq_rank[["Rank", "Floor", "Room", "Priority_Score"]]
    eq_equip_df = build_eq_equipment_table(eq_rank)

    fire_rank = rank_fire_equipment_priorities(
        results["fire_data"],
        results["attributes"],
        rooms_filter=fire_selected_rooms,
    ).copy()
    fire_rank.insert(0, "Rank", np.arange(1, len(fire_rank) + 1))
    fire_rank_simple = fire_rank[["Rank", "Floor", "Room", "Equipment_Priority_Score", "Purpose"]]
    fire_equip_df = build_fire_equipment_table(fire_rank)

    tab_fire, tab_eq, tab_both = st.tabs(["Fire", "Earthquake", "Combined"])

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
        st.dataframe(eq_rank_simple, use_container_width=True, hide_index=True)
        with st.expander("Show details"):
            st.dataframe(eq_rank, use_container_width=True, hide_index=True)

        eq_rank_csv = eq_rank.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Earthquake Ranking (CSV)",
            data=eq_rank_csv,
            file_name="Earthquake_Evacuation_Priorities.csv",
            mime="text/csv",
            key="eq_rank_dl_tab",
        )

        st.subheader("Equipment Recommendations")
        render_gallery_toggle_button("eq")
        render_clickable_equipment_table(eq_equip_df, key_prefix="eq_equip")
        eq_equip_csv = eq_equip_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Equipment Recommendations (CSV)",
            data=eq_equip_csv,
            file_name="Earthquake_Equipment_Recommendations.csv",
            mime="text/csv",
            key="eq_equip_dl_tab",
        )

        st.subheader("Earthquake Scenario Data")
        st.dataframe(results["eq_output"], use_container_width=True)
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
        st.dataframe(fire_rank_simple, use_container_width=True, hide_index=True)
        with st.expander("Show details"):
            st.dataframe(fire_rank, use_container_width=True, hide_index=True)

        st.subheader("Equipment Recommendations")
        render_clickable_equipment_table(fire_equip_df, key_prefix="fire_equip")
        equip_csv = fire_equip_df.to_csv(index=False).encode('utf-8')
        fire_rank_csv = fire_rank.to_csv(index=False).encode('utf-8')
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.download_button(
                label="Save Equipment Recommendations (CSV)",
                data=equip_csv,
                file_name="Fire_Equipment_Recommendations.csv",
                mime="text/csv",
                key="fire_equip_dl_tab",
            )
        with c2:
            render_gallery_toggle_button("fire")
        with c3:
            st.download_button(
                label="Save Fire Equipment Ranking (CSV)",
                data=fire_rank_csv,
                file_name="Fire_Safety_Equipment_Priorities.csv",
                mime="text/csv",
                key="fire_rank_dl_tab",
            )
        
        st.subheader("Fire Scenario Data")
        st.dataframe(results["fire_output"], use_container_width=True)
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
        st.markdown("Earthquake")
        st.dataframe(eq_rank_simple, use_container_width=True, hide_index=True)
        with st.expander("Show details"):
            st.dataframe(eq_rank, use_container_width=True, hide_index=True)
        eq_rank_b_csv = eq_rank.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Earthquake Ranking (CSV)",
            data=eq_rank_b_csv,
            file_name="Earthquake_Evacuation_Priorities.csv",
            mime="text/csv",
            key="eq_rank_dl_combined",
        )

        st.markdown("Fire")
        st.dataframe(fire_rank_simple, use_container_width=True, hide_index=True)
        with st.expander("Show details"):
            st.dataframe(fire_rank, use_container_width=True, hide_index=True)
        fire_rank_b_csv = fire_rank.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Fire Equipment Ranking (CSV)",
            data=fire_rank_b_csv,
            file_name="Fire_Safety_Equipment_Priorities.csv",
            mime="text/csv",
            key="fire_rank_dl_combined",
        )

        st.subheader("Equipment Recommendations")
        st.markdown("Earthquake")
        render_clickable_equipment_table(eq_equip_df, key_prefix="eq_equip_combined")
        eq_equip_b_csv = eq_equip_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Earthquake Equipment Recommendations (CSV)",
            data=eq_equip_b_csv,
            file_name="Earthquake_Equipment_Recommendations.csv",
            mime="text/csv",
            key="eq_equip_dl_combined",
        )

        st.markdown("Fire")
        render_clickable_equipment_table(fire_equip_df, key_prefix="fire_equip_combined")
        fire_equip_b_csv = fire_equip_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Fire Equipment Recommendations (CSV)",
            data=fire_equip_b_csv,
            file_name="Fire_Equipment_Recommendations.csv",
            mime="text/csv",
            key="fire_equip_dl_combined",
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

    # Save CSV outputs to disk and confirm
    if run:
        output_dir = "demo_results"
        os.makedirs(output_dir, exist_ok=True)
        fire_path = os.path.join(output_dir, "Fire_Evacuation_Route.csv")
        eq_path = os.path.join(output_dir, "Earthquake_Evacuation_Route.csv")

        results["fire_output"].to_csv(fire_path, index=False)
        results["eq_output"].to_csv(eq_path, index=False)
        st.success("💾 You can save the results as Fire_Evacuation_Route.csv and Earthquake_Evacuation_Route.csv")
        st.toast("💾 You can save the results as Fire_Evacuation_Route.csv and Earthquake_Evacuation_Route.csv")
if data_mode == "Demo data" and not (run or "demo_results" in st.session_state):
    st.info("Set parameters in the sidebar and click 'Run Simulation'.")

# --- Upload CSV & Prediction (AHP-based) -------------------------------------
if data_mode == "Upload CSV":
    st.divider()
    st.header("Upload CSV and Compute Priorities (AHP)")

    # Configuration Row
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        scenario_choice = st.radio("Select Scenario", ["Fire", "Earthquake"], horizontal=True)
    with col_cfg2:
        distance_mode = st.selectbox("Distance Policy", ["far_first", "near_first"], 
                                    help="Should farther or nearer rooms be prioritized?")

    # Check for upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            cols = list(user_df.columns)

            st.subheader("🛠️ Column Mapping") # Remove emoji if needed

            # Helper for auto-detection
            def detect(targets, options):
                for t in targets:
                    for o in options:
                        if t.lower() == o.lower(): return options.index(o)
                return 0

            # Mapping UI
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                sel_room = st.selectbox("Room ID", cols, index=detect(["Room", "room_label", "Label"], cols))
            with m2:
                sel_floor = st.selectbox("Floor", cols, index=detect(["Floor", "Level"], cols))
            with m3:
                sel_dist = st.selectbox("Distance", cols, index=detect(["Distance", "distance_to_exit_m"], cols))
            with m4:
                risk_default = "fire_risk" if scenario_choice == "Fire" else "quake_risk"
                sel_risk = st.selectbox("Risk (Optional)", ["<none>"] + cols, 
                                        index=detect([risk_default, "Risk", "quake_risk", "fire_risk"], ["<none>"] + cols))

            # Transform data here
            mapping = {sel_room: "room_label", sel_floor: "floor", sel_dist: "distance_to_exit_m"}
            if sel_risk != "<none>":
                risk_internal_name = "fire_risk" if scenario_choice == "Fire" else "quake_risk"
                mapping[sel_risk] = risk_internal_name

            mapped_df = user_df.rename(columns=mapping)
            with st.expander("Preview Mapped Data (Raw)", expanded=True):
                st.dataframe(mapped_df.head(10), use_container_width=True)

            # Computation logic
            if st.button("🚀 Compute Priorities", type="primary"): # Remove emoji if weird
                with st.spinner("Calculating..."):
                    # The backend expects 'floor', 'room_label', etc. in lowercase
                    # A bit confused initially dito
                    priorities_df, meta = core_compute_priorities(
                        mapped_df,
                        scenario=scenario_choice,
                        distance_mode=distance_mode,
                    )

                     # OPTIONAL, pero astig tingnan ngl
                    st.toast(f"✅ Analysis Complete for {scenario_choice} Evacuation Priorities.")
                    
                    # Prepare for display
                    res_df = priorities_df.copy()
                    if "Rank" not in res_df.columns:
                        res_df.insert(0, "Rank", range(1, len(res_df) + 1))

                    # Display Table
                    display_df = priorities_df.copy()

                    # Handle yung rank thingy
                    if "Rank" not in display_df.columns:
                        display_df.insert(0, "Rank", np.arange(1, len(display_df) + 1))

                    # Prettify
                    column_mapping = {
                        "room_label": "Room",
                        "floor": "Floor",
                        "priority": "Priority Score"
                    }

                    # Filter to only the columns that actually exist in the result
                    existing_cols = ["Rank"] + [c for c in column_mapping.keys() if c in display_df.columns]
                    prio_display = display_df[existing_cols].rename(columns=column_mapping)
                    st.dataframe(prio_display, use_container_width=True, hide_index=True)

                    # Visualization logic
                    # Added because it felt empty, remove if necessary
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(res_df["room_label"].astype(str), res_df["priority"], 
                            marker='o', color="purple", linewidth=2)
                    ax.set_title(f"{scenario_choice} Evacuation Priorities")
                    ax.set_ylabel("Priority Score")
                    ax.set_xlabel("Room")
                    plt.xticks(rotation=45)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    st.pyplot(fig)

                    # Download logic
                    csv_data = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Results (CSV)", csv_data, 
                                    f"{scenario_choice}_Priorities.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            st.exception(e) # This will show yung error for sure, NGL IDK there was an st.exception
