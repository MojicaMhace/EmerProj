import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
import json
import re

from evacuation import (
    artificial_bee_colony,
    room_labels as gen_room_labels,
    rank_earthquake_priorities,
    rank_fire_equipment_priorities,
    ahp_weights,
)
from evac_core import demo_data as core_demo_data

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None
    _HAS_OPENAI = False

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

def render_building_layout(
    floors,
    rooms_per_floor,
    results,
    color_mode,
    tile_size,
    fire_selected_rooms,
    eq_selected_rooms,
    key_prefix: str,
):
    try:
        floors = int(floors)
        rooms_per_floor = int(rooms_per_floor)
    except Exception:
        return

    tile_px = max(16, min(80, int(tile_size or 48)))
    color_mode = (color_mode or "None").lower()

    fire_df = results.get("fire_output") if isinstance(results, dict) else None
    eq_df = results.get("eq_output") if isinstance(results, dict) else None
    attrs_df = results.get("attributes") if isinstance(results, dict) else None

    def _maybe_float(val):
        try:
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return None
            return float(val)
        except Exception:
            return None

    def _lookup(df: pd.DataFrame | None, room_id: int, col: str):
        if df is None or col not in df.columns:
            return None
        match = df[df["Room"] == room_id]
        if match.empty:
            return None
        value = match.iloc[0][col]
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    def _normalize_selected(selected):
        out = set()
        for item in selected or []:
            if isinstance(item, str):
                cleaned = item.strip()
                out.add(cleaned)
                if cleaned.lower().startswith("room "):
                    out.add(cleaned.split(" ", 1)[1])
            else:
                try:
                    out.add(str(int(item)))
                except Exception:
                    pass
        return out

    fire_selected = _normalize_selected(fire_selected_rooms)
    eq_selected = _normalize_selected(eq_selected_rooms)

    rooms_payload = []
    for floor in range(1, floors + 1):
        for room_no in range(1, rooms_per_floor + 1):
            room_id = (floor - 1) * rooms_per_floor + room_no
            label_num = f"{floor}{room_no:02d}"
            full_label = f"Room {label_num}"

            fire_priority = _maybe_float(_lookup(fire_df, room_id, "Evacuation_Priority"))
            eq_priority = _maybe_float(_lookup(eq_df, room_id, "Evacuation_Priority"))
            combined_priority = None
            combo_vals = [v for v in (fire_priority, eq_priority) if v is not None]
            if combo_vals:
                combined_priority = float(np.mean(combo_vals))

            if color_mode == "fire":
                priority_for_color = fire_priority
            elif color_mode == "earthquake":
                priority_for_color = eq_priority
            elif color_mode == "combined":
                priority_for_color = combined_priority
            else:
                priority_for_color = None

            risk_val = _maybe_float(_lookup(fire_df if color_mode == "fire" else eq_df, room_id, "Risk"))
            if risk_val is None:
                risk_val = _maybe_float(_lookup(fire_df, room_id, "Risk")) or _maybe_float(_lookup(eq_df, room_id, "Risk"))
            distance_val = _maybe_float(_lookup(fire_df, room_id, "Distance")) or _maybe_float(_lookup(eq_df, room_id, "Distance"))
            occupancy_val = _lookup(attrs_df, room_id, "Occupancy")
            structural_val = _lookup(attrs_df, room_id, "StructuralSafety")
            purpose_val = _lookup(attrs_df, room_id, "Purpose")

            rooms_payload.append({
                "id": room_id,
                "floor": floor,
                "roomNo": room_no,
                "label": label_num,
                "priority": combined_priority,
                "colorPriority": priority_for_color,
                "firePriority": fire_priority,
                "eqPriority": eq_priority,
                "risk": risk_val,
                "distance": distance_val,
                "occupancy": occupancy_val,
                "structural": structural_val,
                "purpose": str(purpose_val) if purpose_val is not None else "",
                "isFire": full_label in fire_selected or label_num in fire_selected,
                "isEq": full_label in eq_selected or label_num in eq_selected,
            })

    rooms_json = json.dumps(rooms_payload)
    wrapper_id = f"layout_wrap_{key_prefix}"
    grid_id = f"layout_grid_{key_prefix}"
    info_id = f"layout_info_{key_prefix}"
    tooltip_id = f"layout_tip_{key_prefix}"
    grid_height = max(260, min(700, tile_px * floors + 120))
    component_height = int(min(900, grid_height + 220))

    html = f"""
    <style>
    #{wrapper_id} {{
        --tile: {tile_px}px;
        --gap: 10px;
        --neutral: #4b5563;
        font-family: "Source Sans Pro", "Inter", system-ui, -apple-system, sans-serif;
        color: #e5e7eb;
    }}
    #{wrapper_id} .card {{
        background: linear-gradient(135deg, #0f172a, #0b1222 55%, #0f172a);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }}
    #{wrapper_id} .building-shell {{
        background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.02), rgba(255,255,255,0));
        border: 2px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }}
    #{wrapper_id} .grid {{
        background: rgba(15,23,42,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 10px;
        overflow: auto;
        max-height: {grid_height}px;
    }}
    #{wrapper_id} .floor {{
        display: grid;
        grid-template-columns: 90px 1fr;
        gap: 12px;
        align-items: center;
        position: relative;
        padding: 12px 8px;
        border-bottom: 1px dashed rgba(255,255,255,0.08);
    }}
    #{wrapper_id} .floor:last-child {{
        border-bottom: none;
    }}
    #{wrapper_id} .floor-label {{
        justify-self: end;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #cbd5e1;
        padding: 6px 10px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}
    #{wrapper_id} .floor-body {{
        position: relative;
        padding: 8px 4px;
    }}
    #{wrapper_id} .corridor {{
        position: absolute;
        left: 0;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        height: calc(var(--tile) / 3);
        background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.12), rgba(255,255,255,0.06));
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06), 0 10px 24px rgba(0,0,0,0.18);
        opacity: 0.9;
    }}
    #{wrapper_id} .row {{
        position: relative;
        display: grid;
        grid-template-columns: repeat({rooms_per_floor}, var(--tile));
        gap: var(--gap);
        align-items: center;
        width: max-content;
        margin: 0 auto;
        z-index: 2;
    }}
    #{wrapper_id} .tile {{
        height: var(--tile);
        width: var(--tile);
        border-radius: 12px;
        background: var(--neutral);
        color: #f8fafc;
        display: grid;
        grid-template-rows: 1fr auto;
        align-items: center;
        justify-items: center;
        font-weight: 700;
        font-size: 0.9rem;
        cursor: pointer;
        position: relative;
        border: 1px solid rgba(255,255,255,0.12);
        transition: transform 0.1s ease, box-shadow 0.1s ease, border-color 0.2s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }}
    #{wrapper_id} .tile:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(0,0,0,0.32);
        border-color: rgba(255,255,255,0.22);
    }}
    #{wrapper_id} .tile .label {{ font-size: 0.95rem; }}
    #{wrapper_id} .tile .meta {{ font-size: 0.72rem; opacity: 0.9; }}
    #{wrapper_id} .tile.active {{ outline: 2px solid #f8fafc; outline-offset: 2px; }}
    #{wrapper_id} .tile.fire {{ box-shadow: 0 0 0 2px #ff9800 inset, 0 10px 22px rgba(0,0,0,0.3); }}
    #{wrapper_id} .tile.eq {{ box-shadow: 0 0 0 2px #4ea1ff inset, 0 10px 22px rgba(0,0,0,0.3); }}
    #{wrapper_id} .tile.both {{ box-shadow: 0 0 0 2px #ff9800 inset, 0 0 0 5px #4ea1ff inset, 0 12px 26px rgba(0,0,0,0.35); }}
    #{wrapper_id} .info {{
        margin-top: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px;
        background: linear-gradient(145deg, rgba(255,255,255,0.02), rgba(255,255,255,0.04));
    }}
    #{wrapper_id} .info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 8px;
        margin-top: 6px;
    }}
    #{wrapper_id} .metric {{
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.06);
    }}
    #{wrapper_id} .metric span {{ font-size: 0.8rem; color: #cbd5e1; }}
    #{wrapper_id} .metric strong {{ font-size: 1rem; color: #f8fafc; }}
    #{wrapper_id} .tooltip {{
        position: fixed;
        pointer-events: none;
        display: none;
        z-index: 9999;
        background: #0b1222;
        color: #e5e7eb;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 16px 32px rgba(0,0,0,0.45);
        min-width: 190px;
        font-size: 12px;
    }}
    #{wrapper_id} .tooltip-title {{ font-weight: 700; margin-bottom: 6px; color: #f9fafb; }}
    #{wrapper_id} .tooltip-grid {{ display: grid; grid-template-columns: repeat(2, minmax(90px, 1fr)); gap: 4px 8px; }}
    #{wrapper_id} .tooltip-grid span {{ color: #cbd5e1; font-size: 11px; }}
    #{wrapper_id} .tooltip-grid strong {{ color: #f1f5f9; font-weight: 700; font-size: 12px; }}
    </style>
    <div id="{wrapper_id}">
      <div class="card">
        <div class="building-shell">
          <div class="grid" id="{grid_id}"></div>
        </div>
        <div class="info" id="{info_id}">Hover or click a room tile to view details.</div>
      </div>
      <div class="tooltip" id="{tooltip_id}"></div>
    </div>
    <script>
    (function() {{
      const rooms = {rooms_json};
      const wrapper = document.getElementById("{wrapper_id}");
      const grid = document.getElementById("{grid_id}");
      const info = document.getElementById("{info_id}");
      const tip = document.getElementById("{tooltip_id}");
      const colorMode = "{color_mode}";
      let activeTile = null;

      function fmt(val, digits=2) {{
        if (val === null || val === undefined || Number.isNaN(val)) return "-";
        if (typeof val === "number") return val.toFixed(digits);
        return val;
      }}

      function colorForScore(score) {{
        if (score === null || score === undefined || Number.isNaN(score)) return getComputedStyle(wrapper).getPropertyValue("--neutral");
        const clamped = Math.max(0, Math.min(1, Number(score)));
        const hue = 120 - 120 * clamped;
        return `hsl(${{hue}}, 70%, 52%)`;
      }}

      function buildTooltip(room) {{
        return `
          <div class="tooltip-title">Floor ${'{'}room.floor{'}'} - Room ${'{'}room.label{'}'}</div>
          <div class="tooltip-grid">
            <div><span>Priority</span><strong>${'{'}fmt(room.priority, 3){'}'}</strong></div>
            <div><span>Risk</span><strong>${'{'}fmt(room.risk, 3){'}'}</strong></div>
            <div><span>Distance</span><strong>${'{'}fmt(room.distance, 2){'}'} m</strong></div>
            <div><span>Occupancy</span><strong>${'{'}fmt(room.occupancy, 0){'}'}</strong></div>
            <div><span>Structural</span><strong>${'{'}fmt(room.structural, 2){'}'}</strong></div>
            <div><span>Purpose</span><strong>${'{'}room.purpose || "-"{'}'}</strong></div>
          </div>`;
      }}

      function renderInfo(room) {{
        const colorLabel = colorMode && colorMode !== "none" ? colorMode : "combined";
        info.innerHTML = `
          <div class="info-grid">
            <div class="metric"><span>Room</span><strong>Floor ${'{'}room.floor{'}'} - ${'{'}room.label{'}'}</strong></div>
            <div class="metric"><span>Priority (${ '{' }colorLabel{ '}' })</span><strong>${'{'}fmt(room.priority, 3){'}'}</strong></div>
            <div class="metric"><span>Risk</span><strong>${'{'}fmt(room.risk, 3){'}'}</strong></div>
            <div class="metric"><span>Distance</span><strong>${'{'}fmt(room.distance, 2){'}'} m</strong></div>
            <div class="metric"><span>Occupancy</span><strong>${'{'}fmt(room.occupancy, 0){'}'}</strong></div>
            <div class="metric"><span>Structural Safety</span><strong>${'{'}fmt(room.structural, 2){'}'}</strong></div>
            <div class="metric"><span>Purpose</span><strong>${'{'}room.purpose || "-"{'}'}</strong></div>
            <div class="metric"><span>Fire Priority</span><strong>${'{'}fmt(room.firePriority, 3){'}'}</strong></div>
            <div class="metric"><span>Earthquake Priority</span><strong>${'{'}fmt(room.eqPriority, 3){'}'}</strong></div>
          </div>`;
      }}

      function hideTip() {{ tip.style.display = "none"; }}
      function showTip(room, evt) {{
        tip.innerHTML = buildTooltip(room);
        tip.style.display = "block";
        // Position tooltip tightly near the cursor
        const x = evt.clientX + 6;
        const y = evt.clientY + 6;
        tip.style.left = x + "px";
        tip.style.top = y + "px";
        tip.style.transform = "translate(0, 0)";
      }}

      const floorsList = Array.from(new Set(rooms.map(r => r.floor))).sort((a,b) => b - a);
      floorsList.forEach(floor => {{
        const floorWrap = document.createElement("div");
        floorWrap.className = "floor";

        const floorLabel = document.createElement("div");
        floorLabel.className = "floor-label";
        floorLabel.textContent = `Floor ${{floor}}`;  // escape python f-string so JS can interpolate
        floorWrap.appendChild(floorLabel);

        const body = document.createElement("div");
        body.className = "floor-body";

        const corridor = document.createElement("div");
        corridor.className = "corridor";
        body.appendChild(corridor);

        const row = document.createElement("div");
        row.className = "row";

        rooms.filter(r => r.floor === floor).sort((a,b) => a.roomNo - b.roomNo).forEach(room => {{
          const tile = document.createElement("div");
          tile.className = "tile";
          if (room.isFire && room.isEq) tile.classList.add("both");
          else if (room.isFire) tile.classList.add("fire");
          else if (room.isEq) tile.classList.add("eq");
          tile.style.background = (colorMode && colorMode !== "none") ? colorForScore(room.colorPriority) : getComputedStyle(wrapper).getPropertyValue("--neutral");
          tile.innerHTML = `<div class="label">${'{'}room.label{'}'}</div><div class="meta">${'{'}room.priority !== null ? fmt(room.priority, 2) : "-"{'}'}</div>`;

          tile.addEventListener("mouseenter", evt => showTip(room, evt));
          tile.addEventListener("mousemove", evt => showTip(room, evt));
          tile.addEventListener("mouseleave", hideTip);
          tile.addEventListener("click", () => {{
            if (activeTile) activeTile.classList.remove("active");
            activeTile = tile;
            tile.classList.add("active");
            renderInfo(room);
          }});

          row.appendChild(tile);
        }});

        body.appendChild(row);
        floorWrap.appendChild(body);
        grid.appendChild(floorWrap);
      }});

      grid.addEventListener("scroll", hideTip);
    }})();</script>
    """

    components.html(html, height=component_height, scrolling=False)

def _parse_room_number(label: str | None) -> int | None:
    if not isinstance(label, str):
        return None
    digits = re.findall(r"\d+", label)
    if not digits:
        return None
    try:
        num = int(digits[-1])
        if num >= 100:
            return num % 100 if num % 100 != 0 else num
        return num
    except Exception:
        return None

def standardize_building(df: pd.DataFrame, rooms_per_floor_hint: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Floor", "Room", "Room_No", "Distance", "fire_risk", "quake_risk", "room_label", "Occupancy", "StructuralSafety", "Purpose"])

    raw = df.copy()
    lower_map = {c.lower(): c for c in raw.columns}

    def pick(*names):
        for name in names:
            if name in lower_map:
                return lower_map[name]
        return None

    floor_col = pick("floor")
    dist_col = pick("distance_to_exit_m", "distance")
    room_label_col = pick("room_label", "label", "room")
    fire_col = pick("fire_risk", "firerisk", "risk_fire")
    quake_col = pick("quake_risk", "quakerisk", "risk_quake", "earthquake_risk")
    occupancy_col = pick("occupancy")
    structural_col = pick("structuralsafety", "structural", "structural_safety")
    purpose_col = pick("purpose")

    if floor_col is None or dist_col is None:
        raise ValueError("Input data must include 'floor' and 'distance_to_exit_m' (or 'distance') columns.")

    floors = raw[floor_col].astype(float).fillna(0).astype(int)
    labels = raw[room_label_col].astype(str) if room_label_col else pd.Series([None] * len(raw))
    distances = raw[dist_col].astype(float).fillna(0.0)

    norm_floor = (floors - floors.min()) / (floors.max() - floors.min() + 1e-9)
    fire_risk = raw[fire_col].astype(float) if fire_col else 1.0 - norm_floor
    quake_risk = raw[quake_col].astype(float) if quake_col else norm_floor

    room_nos = []
    floor_counts: dict[int, int] = {}
    for floor_val, label in zip(floors, labels):
        parsed = _parse_room_number(label)
        if parsed is None:
            parsed = floor_counts.get(floor_val, 0) + 1
        room_nos.append(int(parsed))
        floor_counts[floor_val] = max(floor_counts.get(floor_val, 0), int(parsed))

    rooms_per_floor = int(rooms_per_floor_hint or (max(floor_counts.values()) if floor_counts else 1))
    room_ids = [(int(f) - 1) * rooms_per_floor + int(rn) for f, rn in zip(floors, room_nos)]

    canonical_labels = []
    for f, rn, label in zip(floors, room_nos, labels):
        if isinstance(label, str) and label.strip():
            canonical_labels.append(label)
        else:
            canonical_labels.append(f"Room {int(f)}{int(rn):02d}")

    rng = np.random.default_rng(42)
    occupancy = raw[occupancy_col].fillna(0).astype(float) if occupancy_col else (rng.integers(5, 50, size=len(raw)) * (1 + (floors / max(floors)) * 0.1)).astype(int)
    structural = np.clip(raw[structural_col].astype(float).fillna(0.6), 0.0, 1.0) if structural_col else np.clip(
        0.9 - (floors - floors.min()) / (floors.max() - floors.min() + 1e-9) * 0.3 + rng.normal(0, 0.02, len(raw)),
        0.3,
        1.0,
    )
    if purpose_col:
        purpose = raw[purpose_col].fillna("Office").astype(str)
    else:
        purpose_choices = ["Office", "Meeting", "Lab", "Storage", "Kitchen", "Server"]
        purpose_probs = [0.35, 0.2, 0.1, 0.15, 0.1, 0.1]
        purpose = pd.Series(rng.choice(purpose_choices, size=len(raw), p=purpose_probs))

    return pd.DataFrame({
        "Floor": floors.astype(int),
        "Room": pd.Series(room_ids).astype(int),
        "Room_No": pd.Series(room_nos).astype(int),
        "Distance": distances,
        "fire_risk": pd.Series(fire_risk).astype(float),
        "quake_risk": pd.Series(quake_risk).astype(float),
        "room_label": pd.Series(canonical_labels),
        "Occupancy": pd.Series(occupancy).astype(float),
        "StructuralSafety": pd.Series(structural).astype(float),
        "Purpose": pd.Series(purpose),
    })

def compute_priorities(df: pd.DataFrame,
                       scenario: str,
                       risk_w: float,
                       dist_w: float,
                       intensity: float,
                       abc_params: dict,
                       distance_mode: str = "far_first"):
    if df is None or df.empty:
        return pd.DataFrame(), 0.0

    scenario_key = scenario.lower().strip()
    risk_col = "fire_risk" if scenario_key.startswith("fire") else "quake_risk"
    work = df.copy()
    if risk_col not in work.columns:
        fallback = "fire_risk" if risk_col == "quake_risk" else "quake_risk"
        work[risk_col] = work.get(fallback, 0.5)

    risk_vals = work[risk_col].astype(float).values
    if scenario_key.startswith("earthquake") and intensity is not None:
        try:
            scale = 1.0 + 0.05 * max(0.0, float(intensity))
            risk_vals = np.clip(risk_vals * scale, 0.0, None)
        except Exception:
            pass

    dist_vals = work["Distance"].astype(float).values
    risk_norm = (risk_vals - risk_vals.min()) / (risk_vals.max() - risk_vals.min() + 1e-9)
    dist_norm = (dist_vals - dist_vals.min()) / (dist_vals.max() - dist_vals.min() + 1e-9)
    if distance_mode == "near_first":
        dist_norm = 1.0 - dist_norm

    # Normalize weights so higher weight emphasizes higher priority
    w_sum = float(risk_w) + float(dist_w) + 1e-9
    w_risk = float(risk_w) / w_sum
    w_dist = float(dist_w) / w_sum

    data_two_cols = np.column_stack([risk_norm, dist_norm])
    ratio = max(1e-6, float(risk_w) / float(dist_w or 1.0))
    w_matrix = np.array([[1.0, ratio], [1.0 / ratio, 1.0]])
    weights = ahp_weights(w_matrix)

    params = {
        "colony_size": int(abc_params.get("colony_size", 20)),
        "max_iter": int(abc_params.get("max_iter", 50)),
        "limit": int(abc_params.get("limit", 10)),
        "seed": abc_params.get("seed", 42),
    }

    solution, fitness = artificial_bee_colony(
        data_two_cols,
        weights,
        colony_size=params["colony_size"],
        max_iter=params["max_iter"],
        limit=params["limit"],
        seed=params["seed"],
    )
    priority = np.clip(w_risk * risk_norm + w_dist * dist_norm, 0.0, 1.0)

    out = pd.DataFrame({
        "Floor": work["Floor"].astype(int),
        "Room": work["Room"].astype(int),
        "Risk": risk_vals,
        "Distance": dist_vals,
        "Evacuation_Priority": priority,
        "room_label": work.get("room_label", pd.Series([f"Room {r}" for r in work["Room"]])),
    })

    return out, float(fitness)

def build_results(input_df: pd.DataFrame,
                  rooms_per_floor_hint: int,
                  fire_risk_w: float,
                  fire_dist_w: float,
                  eq_risk_w: float,
                  eq_dist_w: float,
                  intensity: float,
                  abc_params: dict,
                  distance_mode: str = "far_first") -> dict:
    standardized = standardize_building(input_df, rooms_per_floor_hint=rooms_per_floor_hint)

    fire_output, fire_fit = compute_priorities(
        standardized, "fire", fire_risk_w, fire_dist_w, intensity, abc_params, distance_mode
    )
    eq_output, eq_fit = compute_priorities(
        standardized, "earthquake", eq_risk_w, eq_dist_w, intensity, abc_params, distance_mode
    )

    attributes = standardized[["Room", "Occupancy", "StructuralSafety", "Purpose"]].copy()
    fire_data = standardized[["Floor", "Room", "fire_risk", "Distance", "Purpose"]].rename(columns={"fire_risk": "Risk"})
    eq_data = standardized[["Floor", "Room", "quake_risk", "Distance"]].rename(columns={"quake_risk": "Risk"})

    return {
        "attributes": attributes,
        "fire_output": fire_output,
        "eq_output": eq_output,
        "fire_best_fitness": fire_fit,
        "eq_best_fitness": eq_fit,
        "fire_data": fire_data,
        "earthquake_data": eq_data,
        "room_labels": standardized["room_label"].tolist(),
    }

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
layout_container = st.container()

with st.sidebar:
    st.header("Parameters")
    data_mode = st.radio(
        "Data source",
        options=["Upload CSV (Realistic)", "Demo data (Sample only)"],
        index=0,
    )
    if data_mode == "Demo data (Sample only)":
        st.caption("Demo data is simulated and may not reflect real buildings.")
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

    run = st.button("Run / Compute", type="primary")

is_demo = data_mode == "Demo data (Sample only)"
abc_params = {
    "colony_size": int(colony_size),
    "max_iter": int(max_iter),
    "limit": int(limit),
    "seed": 42,
}

results = st.session_state.get("demo_results") if is_demo else st.session_state.get("csv_results")

if is_demo and run:
    demo_df = core_demo_data(floors=int(floors), rooms_per_floor=int(rooms_per_floor), seed=42)
    results = build_results(
        demo_df,
        rooms_per_floor_hint=int(rooms_per_floor),
        fire_risk_w=float(fire_risk_importance),
        fire_dist_w=float(fire_distance_importance),
        eq_risk_w=float(eq_risk_importance),
        eq_dist_w=float(eq_distance_importance),
        intensity=float(intensity),
        abc_params=abc_params,
        distance_mode="far_first",
    )
    st.session_state["demo_results"] = results

# --- Upload CSV (Unified) ----------------------------------------------------
if not is_demo:
    st.divider()
    st.header("Upload CSV and Compute Priorities (AHP)")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        scenario_choice = st.radio("Select Scenario", ["Fire", "Earthquake"], horizontal=True)
    with col_cfg2:
        distance_mode = st.selectbox(
            "Distance Policy",
            ["far_first", "near_first"],
            help="Should farther or nearer rooms be prioritized?",
        )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            cols = list(user_df.columns)

            st.subheader("Column Mapping")

            def detect(targets, options):
                for t in targets:
                    for o in options:
                        if t.lower() == o.lower():
                            return options.index(o)
                return 0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                sel_room = st.selectbox("Room ID", cols, index=detect(["Room", "room_label", "Label"], cols))
            with m2:
                sel_floor = st.selectbox("Floor", cols, index=detect(["Floor", "Level"], cols))
            with m3:
                sel_dist = st.selectbox("Distance", cols, index=detect(["Distance", "distance_to_exit_m"], cols))
            with m4:
                risk_default = "fire_risk" if scenario_choice == "Fire" else "quake_risk"
                sel_risk = st.selectbox(
                    "Risk (Optional)",
                    ["<none>"] + cols,
                    index=detect([risk_default, "Risk", "quake_risk", "fire_risk"], ["<none>"] + cols),
                )

            mapping = {sel_room: "room_label", sel_floor: "floor", sel_dist: "distance_to_exit_m"}
            if sel_risk != "<none>":
                mapping[sel_risk] = "fire_risk" if scenario_choice == "Fire" else "quake_risk"

            mapped_df = user_df.rename(columns=mapping)
            with st.expander("Preview Mapped Data (Raw)", expanded=True):
                st.dataframe(mapped_df.head(10), use_container_width=True)

            if run:
                results = build_results(
                    mapped_df,
                    rooms_per_floor_hint=int(rooms_per_floor),
                    fire_risk_w=float(fire_risk_importance),
                    fire_dist_w=float(fire_distance_importance),
                    eq_risk_w=float(eq_risk_importance),
                    eq_dist_w=float(eq_distance_importance),
                    intensity=float(intensity),
                    abc_params=abc_params,
                    distance_mode=distance_mode,
                )
                st.session_state["csv_results"] = results
                st.toast(f"Analysis Complete for {scenario_choice} Evacuation Priorities.")
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.exception(e)
    else:
        if run:
            st.warning("Upload a CSV file, then click Run / Compute to calculate evacuation priorities.")
        else:
            st.warning("Upload a CSV file to compute evacuation priorities.")

with layout_container:
    st.subheader("Building Layout")
    layout_col1, layout_col2 = st.columns([2, 1])
    with layout_col1:
        layout_color_mode = st.selectbox(
            "Layout color by",
            options=["None", "Fire", "Earthquake", "Combined"],
            index=3,
            help="Color tiles using priority scores from the chosen scenario.",
        )
    with layout_col2:
        layout_tile_size = st.slider("Tile size", min_value=16, max_value=80, value=48, step=2)

    if results is not None:
        render_building_layout(
            floors=floors,
            rooms_per_floor=rooms_per_floor,
            results=results,
            color_mode=layout_color_mode,
            tile_size=layout_tile_size,
            fire_selected_rooms=fire_selected_rooms,
            eq_selected_rooms=eq_selected_rooms,
            key_prefix="demo_layout" if is_demo else "upload_layout",
        )
    elif is_demo:
        st.info("Set parameters in the sidebar and click 'Run / Compute'.")
    else:
        st.warning("Upload a CSV and click 'Run / Compute' to visualize the layout.")

if results is not None:

    st.success("Computation complete.")

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

    fire_data_for_rank = results["fire_data"]
    if isinstance(fire_data_for_rank, pd.DataFrame) and "Purpose" in fire_data_for_rank.columns:
        fire_data_for_rank = fire_data_for_rank.drop(columns=["Purpose"])
    fire_rank = rank_fire_equipment_priorities(
        fire_data_for_rank,
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

    st.subheader("AI Evacuation Summary (optional)")
    api_key = st.text_input("OpenAI API key", type="password", value="sk-REPLACE_ME_9f3a1c", key="ai_api_key")
    generate_summary = st.button("Generate AI explanation")

    if generate_summary:
        st.session_state.pop("ai_summary", None)
        if "REPLACE_ME" in api_key:
            st.warning("Please enter your OpenAI API key to generate the explanation.")
        elif not _HAS_OPENAI:
            st.error("OpenAI SDK not installed. Add `openai` to requirements.txt and reinstall the environment.")
        else:
            # Determine scenario from layout selection; fallback to Combined
            scenario_label = layout_color_mode if layout_color_mode and layout_color_mode.lower() != "none" else "Combined"
            scenario_key = scenario_label.lower()

            fire_df = results["fire_output"][["Floor", "Room", "Evacuation_Priority", "Risk", "Distance"]].copy()
            eq_df = results["eq_output"][["Floor", "Room", "Evacuation_Priority", "Risk", "Distance"]].copy()

            def _safe_num(val):
                try:
                    if pd.isna(val):
                        return None
                except Exception:
                    pass
                try:
                    return float(val)
                except Exception:
                    return None

            if scenario_key.startswith("fire"):
                chosen = fire_df.rename(columns={"Evacuation_Priority": "Priority"})
                risk_w = float(fire_risk_importance)
                dist_w = float(fire_distance_importance)
            elif scenario_key.startswith("earth"):
                chosen = eq_df.rename(columns={"Evacuation_Priority": "Priority"})
                risk_w = float(eq_risk_importance)
                dist_w = float(eq_distance_importance)
            else:
                merged = fire_df.rename(columns={
                    "Evacuation_Priority": "Evacuation_Priority_fire",
                    "Risk": "Risk_fire",
                    "Distance": "Distance_fire",
                }).merge(
                    eq_df.rename(columns={
                        "Evacuation_Priority": "Evacuation_Priority_eq",
                        "Risk": "Risk_eq",
                        "Distance": "Distance_eq",
                    }),
                    on=["Floor", "Room"],
                    how="outer",
                )
                merged["Priority"] = merged[["Evacuation_Priority_fire", "Evacuation_Priority_eq"]].mean(axis=1)
                merged["Risk"] = merged[["Risk_fire", "Risk_eq"]].mean(axis=1)
                merged["Distance"] = merged[["Distance_fire", "Distance_eq"]].mean(axis=1)
                chosen = merged[["Floor", "Room", "Priority", "Risk", "Distance"]]
                risk_w = float(fire_risk_importance + eq_risk_importance) / 2.0
                dist_w = float(fire_distance_importance + eq_distance_importance) / 2.0

            attrs = results.get("attributes")
            if isinstance(attrs, pd.DataFrame):
                chosen = chosen.merge(attrs[["Room", "Occupancy", "Purpose"]], on="Room", how="left")

            top_rows = chosen.sort_values("Priority", ascending=False).head(10)
            payload_records = []
            for _, row in top_rows.iterrows():
                payload_records.append({
                    "floor": int(row["Floor"]) if _safe_num(row["Floor"]) is not None else None,
                    "room": int(row["Room"]) if _safe_num(row["Room"]) is not None else None,
                    "priority": _safe_num(row["Priority"]),
                    "risk": _safe_num(row["Risk"]),
                    "distance": _safe_num(row["Distance"]),
                    "occupancy": _safe_num(row.get("Occupancy")),
                    "purpose": str(row.get("Purpose") or ""),
                })

            current_distance_mode = "far_first"
            try:
                current_distance_mode = distance_mode
            except Exception:
                pass

            payload = {
                "scenario": scenario_label,
                "weights": {
                    "risk_importance": risk_w,
                    "distance_importance": dist_w,
                    "distance_mode": current_distance_mode,
                },
                "top_rooms": payload_records,
            }

            try:
                client = OpenAI(api_key=api_key)
                resp = client.responses.create(
                    model="gpt-5.2",
                    instructions="You are a safety planner. Explain clearly for non-technical users.",
                    input=f"""Explain the evacuation results in simple words.
Include: top 5 rooms to evacuate first and why, which floors are most critical, what to do in the first 3 minutes, and a short disclaimer. Data: {json.dumps(payload)}""",
                )
                summary_text = getattr(resp, "output_text", None) or str(resp)
                st.session_state["ai_summary"] = summary_text
            except Exception as e:
                st.error(f"Failed to generate explanation: {e}")

    if st.session_state.get("ai_summary"):
        st.markdown(st.session_state["ai_summary"])

    # Save CSV outputs to disk and confirm
    if is_demo and run:
        output_dir = "demo_results"
        os.makedirs(output_dir, exist_ok=True)
        fire_path = os.path.join(output_dir, "Fire_Evacuation_Route.csv")
        eq_path = os.path.join(output_dir, "Earthquake_Evacuation_Route.csv")

        results["fire_output"].to_csv(fire_path, index=False)
        results["eq_output"].to_csv(eq_path, index=False)
        st.success("💾 You can save the results as Fire_Evacuation_Route.csv and Earthquake_Evacuation_Route.csv")
        st.toast("💾 You can save the results as Fire_Evacuation_Route.csv and Earthquake_Evacuation_Route.csv")
if is_demo and results is None:
    st.info("Set parameters in the sidebar and click 'Run / Compute'.")

