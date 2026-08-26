"""
UI Helper Module for Optimizer Visualizer
Provides custom CSS injection, telemetry strip component, and Matplotlib figure styling
to maintain a cohesive 'instrument panel' visual identity across Streamlit tabs.
"""

import matplotlib.pyplot as plt
import streamlit as st


def inject_custom_css():
    """Inject global CSS rules for instrument panel styling."""
    css = """
    <style>
    /* -------------------------------------------------------------------------
       Google Fonts Import
       ------------------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* -------------------------------------------------------------------------
       Global Typography & Background
       ------------------------------------------------------------------------- */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headings: Space Grotesk */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeader"],
    .stHeading,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.03em !important;
    }

    /* Sidebar Section Headers & Labels (Small-caps muted style) */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-size: 0.8rem !important;
        color: #8B96A3 !important;
        font-weight: 600 !important;
    }

    /* Metric values & Monospace readouts */
    [data-testid="stMetricValue"],
    .monospace-text,
    .telemetry-strip {
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.75rem;
        color: #8B96A3;
    }

    /* -------------------------------------------------------------------------
       Containers, Cards & Bezels (1px hairline #2E3742 border, #1A2029 bg)
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #13171F !important;
        border-right: 1px solid #2E3742 !important;
    }

    .stExpander,
    [data-testid="stExpander"],
    div[data-testid="stForm"],
    div.stContainer {
        border: 1px solid #2E3742 !important;
        background-color: #1A2029 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }

    .stExpanderDetails {
        background-color: #13171F !important;
    }

    /* -------------------------------------------------------------------------
       Primary Buttons (#E3A23B Amber Accent)
       ------------------------------------------------------------------------- */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    div[data-testid="stSidebar"] button[kind="primary"],
    div[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background-color: #E3A23B !important;
        color: #10141A !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        border: 1px solid #E3A23B !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: background-color 0.15s ease-in-out;
    }

    button[kind="primary"] p,
    button[data-testid="baseButton-primary"] p,
    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="baseButton-primary"] p {
        color: #10141A !important;
        font-weight: 700 !important;
    }

    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #F0B34B !important;
        color: #10141A !important;
        border-color: #F0B34B !important;
        box-shadow: none !important;
    }

    /* Secondary buttons */
    .stButton > button:not([data-testid="baseButton-primary"]) {
        border: 1px solid #2E3742 !important;
        background-color: #1A2029 !important;
        color: #E7EAEE !important;
        border-radius: 4px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .stButton > button:not([data-testid="baseButton-primary"]):hover {
        border-color: #8B96A3 !important;
        background-color: #242C37 !important;
        color: #FFFFFF !important;
    }

    /* -------------------------------------------------------------------------
       Horizontal Hairline Dividers
       ------------------------------------------------------------------------- */
    hr,
    [data-testid="stSidebar"] hr,
    .stMarkdown hr {
        border: none !important;
        border-top: 1px solid #2E3742 !important;
        margin: 1rem 0 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_telemetry_strip(items: dict):
    """
    Render a single-line telemetry strip component with instrument readout aesthetics.
    
    Parameters:
        items (dict): Key-value pairs e.g. {"ITER": "247/500", "LOSS": "0.0412", "GRAD NORM": "1.83"}
    """
    parts = []
    for key, val in items.items():
        parts.append(
            f'<span style="color: #8B96A3; font-family: \'Space Grotesk\', sans-serif; '
            f'text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; '
            f'font-weight: 600;">{key}</span> '
            f'<span style="color: #E3A23B; font-family: \'JetBrains Mono\', monospace; '
            f'font-weight: 700; font-size: 0.88rem; margin-left: 0.25rem; '
            f'margin-right: 0.75rem;">{val}</span>'
        )

    separator = '<span style="color: #2E3742; margin-right: 0.75rem;">|</span>'
    strip_content = separator.join(parts)

    html = f"""
    <div style="
        background-color: #13171F;
        border: 1px solid #2E3742;
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    ">
        {strip_content}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def apply_instrument_theme(fig, ax, legend=None):
    """
    Apply instrument panel styling to a Matplotlib figure and axis.
    
    Parameters:
        fig: Matplotlib Figure object
        ax: Matplotlib Axes object (or list/dict of axes)
        legend: Matplotlib Legend object (optional)
    """
    bg_color = "#1A2029"
    text_color = "#E7EAEE"
    grid_color = "#2E3742"

    fig.patch.set_facecolor(bg_color)

    axes_list = [ax] if not isinstance(ax, (list, tuple, dict)) else (ax.values() if isinstance(ax, dict) else ax)

    for a in axes_list:
        a.set_facecolor(bg_color)
        a.title.set_color(text_color)
        a.xaxis.label.set_color(text_color)
        a.yaxis.label.set_color(text_color)

        a.tick_params(colors=text_color, which="both", labelsize=8)

        for spine in a.spines.values():
            spine.set_color(grid_color)
            spine.set_linewidth(1.0)

        a.grid(True, color=grid_color, linestyle="--", alpha=0.4, linewidth=0.7)

        # Style legend if present on axis
        leg = a.get_legend()
        if leg:
            leg.get_frame().set_facecolor("#10141A")
            leg.get_frame().set_edgecolor("#2E3742")
            leg.get_frame().set_linewidth(0.8)
            for text in leg.get_texts():
                text.set_color(text_color)
