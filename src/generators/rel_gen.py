"""
Pareto Chart Generator
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker


# ============================================================================
# CONSTANTS
# ============================================================================

# Category color mapping (updated to AI/TS terminology)
CATEGORY_COLORS = {
    # Full surrogate comparisons
    "AI/AI": "#1f77b4", "AI/TS": "#f48c06", "AI/SC": "#6f2dbd",
    "TS/TS": "#1f77b4", "TS/AI": "#8c564b", "TS/SC": "#17becf",
    "SC/SC": "#1f77b4", "SC/AI": "#6f2dbd", "SC/TS": "#17becf",

    # Partial surrogate comparisons
    "SC+AI/SC+AI": "#ff2b8a", "SC+AI/SC+TS": "#6f2dbd", "SC+AI/SC": "#17becf",
    "SC+TS/SC+TS": "#1f77b4", "SC+TS/SC+AI": "#8c564b", "SC+TS/SC": "#6f2dbd",
    "SC/SC+AI": "#f48c06", "SC/SC+TS": "#17becf",
}

# Quadrant colors
QUADRANT_COLORS = {
    "tl": (214/255, 233/255, 198/255, 0.4),  # Win-Win (green)
    "tr": (252/255, 243/255, 207/255, 0.4),  # Performance up, Cost up (yellow)
    "bl": (205/255, 219/255, 238/255, 0.4),  # Cost down, Performance down (blue)
    "br": (255/255, 220/255, 220/255, 0.4),  # Lose-Lose (red)
}

FONT_SIZES = {
    "tick": 11,
    "axis": 12,
    "title": 14,
    "legend": 9,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load the processed CSV data produced by the processor."""
    paths = [
        Path("data/processed/processed_db.csv"),
        Path("../data/processed/processed_db.csv"),
        Path("../../data/processed/processed_db.csv"),
    ]

    for path in paths:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(f"Could not find processed_db.csv in: {paths}")


def rebase_comparisons(
    df: pd.DataFrame,
    base: str,
    domain: str,
    scope: Literal["replacement", "hybrid"]
) -> pd.DataFrame:
    """Rebase comparisons to specified baseline."""
    # Filter by domain (skip if data is already filtered for custom clusters)
    if domain == "All Domains":
        filtered = df.copy()
    elif domain.endswith("Cluster") or domain in ["Physical Modeling Sciences", "Earth & Space Sciences", "Life Sciences", "Engineering", "Environmental & Agricultural Sciences", "Economic Sciences", "Other"]:
        # Custom cluster data is already filtered
        filtered = df.copy()
    else:
        filtered = df[
            (df["paper_domain"] == domain) | 
            (df["paper_granular_domain"] == domain)
        ].copy()
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Filter by scope using structural heuristic on comparison_type
    if scope == "replacement":
        mask = filtered["comparison_type"].apply(
            lambda ct: isinstance(ct, str) and "/" in ct and "+" not in ct
        )
    else:  # hybrid
        mask = filtered["comparison_type"].apply(
            lambda ct: isinstance(ct, str) and "/" in ct and any("+" in s for s in ct.split("/"))
        )
    
    filtered = filtered[mask]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Keep only needed columns and drop NaNs
    filtered = filtered[["paper_id", "comparison_type", "relative_cost", "relative_perf"]].dropna()
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Rebase to the specified baseline
    rebased = []
    base_lower = base.lower()
    
    for _, row in filtered.iterrows():
        ct = row["comparison_type"]
        if "/" not in ct:
            continue
        
        left, right = ct.split("/")
        left = left.strip()
        right = right.strip()
        
        # Check if this comparison involves the baseline
        if left.lower() == base_lower:
            # Already in correct orientation
            rebased.append({
                "paper_id": row["paper_id"],
                "comparison_type": f"{left}/{right}",
                "relative_cost": row["relative_cost"],
                "relative_perf": row["relative_perf"]
            })
        elif right.lower() == base_lower:
            # Need to flip the comparison
            rebased.append({
                "paper_id": row["paper_id"],
                "comparison_type": f"{right}/{left}",
                "relative_cost": -row["relative_cost"],
                "relative_perf": -row["relative_perf"]
            })
    
    return pd.DataFrame(rebased)


# ============================================================================
# CHART DRAWING
# ============================================================================

def draw_background(ax, xlim, ylim):
    """Draw quadrant backgrounds and labels."""
    x0, x1 = xlim
    y0, y1 = ylim
    
    # Draw quadrants
    if x0 < 0 and y1 > 0:  # Top-left
        ax.add_patch(Rectangle((x0, max(0, y0)), min(0, x1)-x0, y1-max(0, y0),
                              facecolor=QUADRANT_COLORS['tl'], edgecolor="none", zorder=0))
    
    if x1 > 0 and y1 > 0:  # Top-right
        ax.add_patch(Rectangle((max(0, x0), max(0, y0)), x1-max(0, x0), y1-max(0, y0),
                              facecolor=QUADRANT_COLORS['tr'], edgecolor="none", zorder=0))
    
    if x0 < 0 and y0 < 0:  # Bottom-left
        ax.add_patch(Rectangle((x0, y0), min(0, x1)-x0, min(0, y1)-y0,
                              facecolor=QUADRANT_COLORS['bl'], edgecolor="none", zorder=0))
    
    if x1 > 0 and y0 < 0:  # Bottom-right
        ax.add_patch(Rectangle((max(0, x0), y0), x1-max(0, x0), min(0, y1)-y0,
                              facecolor=QUADRANT_COLORS['br'], edgecolor="none", zorder=0))
    
    # Grid and axes
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.3, color="#888", zorder=0.2)
    ax.axvline(0, color="#888", lw=1.2, zorder=0.3)
    ax.axhline(0, color="#888", lw=1.2, zorder=0.3)
    
    # Quadrant labels
    ax.text(x0 + 0.06*(x1-x0), y1 - 0.04*(y1-y0), "WIN-WIN", 
           color="#2d6a4f", fontsize=11, fontweight="bold", ha="center", va="center",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
    
    ax.text(x1 - 0.07*(x1-x0), y0 + 0.04*(y1-y0), "LOSE-LOSE",
           color="#c52233", fontsize=11, fontweight="bold", ha="center", va="center",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
    
    # Bottom-left label
    bl_x, bl_y = x0 + 0.15*(x1-x0), y0 + 0.04*(y1-y0)
    perf_text = TextArea("EFFICIENCY-PRIORITIZATION", textprops=dict(color="#5a7fa3", fontsize=11, fontweight="bold"))
    vbox = VPacker(children=[perf_text], align="center", pad=0, sep=2)
    ab = AnnotationBbox(vbox, (bl_x, bl_y), frameon=True, zorder=5,
                       bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.add_artist(ab)
    
    # Top-right label  
    tr_x, tr_y = x1 - 0.16*(x1-x0), y1 - 0.04*(y1-y0)
    perf_text = TextArea("PERFORMANCE-PRIORITIZATION", textprops=dict(color="#b8860b", fontsize=11, fontweight="bold"))
    vbox = VPacker(children=[perf_text], align="center", pad=0, sep=2)
    ab = AnnotationBbox(vbox, (tr_x, tr_y), frameon=True, zorder=5,
                       bboxprops=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.add_artist(ab)


def format_ticks(ticks):
    """Format tick labels for log10 scale."""
    labels = []
    for t in ticks:
        if np.isclose(t, 0.0):
            labels.append("SAME")
        elif float(t).is_integer():
            labels.append(f"$10^{{{int(t)}}}$")
        else:
            # Format with appropriate decimals - preserve decimal values
            if abs(t) >= 1:
                # For values >= 1, use 1 decimal place to preserve .5 values
                formatted = f"{t:.1f}".rstrip('0').rstrip('.')
            elif abs(t) >= 0.1:
                formatted = f"{t:.1f}".rstrip('0').rstrip('.')
            else:
                formatted = f"{t:.2f}".rstrip('0').rstrip('.')
            labels.append(f"$10^{{{formatted}}}$")
    return labels


def format_decimal_ticks(ticks):
    """Format tick labels for decimal scale - FIXED VERSION."""
    labels = []
    for t in ticks:
        # Round to avoid floating point precision issues
        t_rounded = round(t, 2)
        
        if np.isclose(t_rounded, 0.0):
            labels.append("SAME")
        else:
            # Use integer formatting for whole numbers, one decimal for half-steps
            if t_rounded == int(t_rounded):
                labels.append(f"{int(t_rounded)}")
            else:
                labels.append(f"{t_rounded:.1f}")
    return labels


def generate_chart(
    domain: str,
    base: str,
    scope: Literal["replacement", "hybrid"],
    include_same: bool = True,
    save_path: Optional[str] = None,
    custom_data: Optional[pd.DataFrame] = None
) -> plt.Figure:
    """Generate a single Pareto chart - only AI baseline by default for replacement scope."""
    # Skip non-AI baselines for replacement scope unless explicitly needed
    if scope == "replacement" and base != "AI":
        return None
    
    # Load and process data
    if custom_data is not None:
        data = rebase_comparisons(custom_data, base, domain, scope)
    else:
        df = load_data()
        data = rebase_comparisons(df, base, domain, scope)
    
    if data.empty:
        return None
    
    # Determine comparison types to show
    if scope == "replacement":
        if base == "AI":
            types = ["AI/AI", "AI/TS", "AI/SC"] if include_same else ["AI/TS", "AI/SC"]
        elif base == "TS":
            types = ["TS/TS", "TS/AI", "TS/SC"] if include_same else ["TS/AI", "TS/SC"]
        else:  # SC
            types = ["SC/SC", "SC/AI", "SC/TS"] if include_same else ["SC/AI", "SC/TS"]
    else:  # hybrid
        if base == "SC":
            types = ["SC/SC", "SC/SC+AI", "SC/SC+TS"] if include_same else ["SC/SC+AI", "SC/SC+TS"]
        elif base == "SC+TS":
            types = ["SC+TS/SC+TS", "SC+TS/SC+AI", "SC+TS/SC"] if include_same else ["SC+TS/SC+AI", "SC+TS/SC"]
        else:  # SC+AI
            types = ["SC+AI/SC+AI", "SC+AI/SC+TS", "SC+AI/SC"] if include_same else ["SC+AI/SC+TS", "SC+AI/SC"]
    
    # Filter to types that exist in data
    available = data["comparison_type"].unique()
    types = [t for t in types if t in available]
    
    if not types:
        return None
    
    # Calculate statistics
    stats = {}
    for comp_type in types:
        comp_data = data[data["comparison_type"] == comp_type]
        if not comp_data.empty:
            stats[comp_type] = {
                "mean": (comp_data["relative_cost"].mean(), comp_data["relative_perf"].mean()),
                "count": len(comp_data),
                "papers": comp_data["paper_id"].nunique()
            }
    
    if not stats:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate axis limits
    x_vals = [s["mean"][0] for s in stats.values()]
    y_vals = [s["mean"][1] for s in stats.values()]
    
    # Set fixed xlim for AI baseline charts
    if base == "AI":
        # Special case for All Domains DL chart
        if domain == "All Domains":
            xlim = (-3.1, 3.1)
        else:
            xlim = (-1.9, 6.1)
    else:
        x_range = max(x_vals) - min(x_vals) if x_vals else 1
        x_pad = max(0.5, x_range * 0.2)
        xlim = (min(x_vals + [-0.1]) - x_pad, max(x_vals + [0.1]) + x_pad)
        # Ensure reasonable limits
        xlim = (max(xlim[0], -10), min(xlim[1], 10))
    
    # Set fixed ylim for all charts
    ylim = (-0.9, 0.9)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Draw background
    draw_background(ax, xlim, ylim)
    
    # Draw vectors
    for comp_type, info in stats.items():
        color = CATEGORY_COLORS.get(comp_type, "#333")
        ax.annotate("", xy=info["mean"], xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", lw=2.5, color=color, alpha=0.8))
    
    # Legend
    handles = [plt.Line2D([0], [0], color=CATEGORY_COLORS.get(c, "#333"), lw=3) for c in stats]
    labels = [f"{c}\n({s['count']} comps, {s['papers']} papers)" for c, s in stats.items()]
    legend = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                      ncol=3, frameon=False, fontsize=FONT_SIZES["legend"])
    
    for text, comp in zip(legend.get_texts(), stats.keys()):
        text.set_ha('center')
        text.set_color(CATEGORY_COLORS.get(comp, "#333"))
    
    # Ticks
    xticks = np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 0.5, 
                      1.0 if xlim[1] - xlim[0] > 5 else 0.5)
    xticks = xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])]
    if 0 not in xticks and xlim[0] <= 0 <= xlim[1]:
        xticks = np.sort(np.append(xticks, 0))
    
    # Generate y-ticks with 0.2 intervals, centered around 0
    yticks = np.arange(-0.8, 0.9, 0.2)
    yticks = yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(format_ticks(xticks), fontsize=FONT_SIZES["tick"])
    ax.set_yticks(yticks)
    # Use same 10^{} format for y-axis as x-axis
    ax.set_yticklabels(format_ticks(yticks), fontsize=FONT_SIZES["tick"])
    
    # Labels
    ax.set_xlabel("Relative Cost", fontsize=FONT_SIZES["axis"], labelpad=10)
    ax.set_ylabel("Relative\nPerformance", fontsize=FONT_SIZES["axis"], 
                 rotation=0, ha="center", y=1.04)
    
    # Title - Replace domain names for better readability
    display_domain = domain.replace("Interdisciplinary", "Other Systems").replace("Engineered Systems", "Engineering Systems")
    scope_text = " (Full Surrogates)" if scope == "full" else " (Partial Surrogates)"
    same_text = " with Same-Base" if include_same else " without Same-Base"
    title = f"{display_domain} \n Cost vs. Performance Trade-offs"
    ax.set_title(title, pad=40, fontsize=FONT_SIZES["title"], fontweight="bold")
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.subplots_adjust(top=0.8, bottom=0.1)
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def generate_custom_cluster_charts(df: pd.DataFrame, custom_clusters: dict, output_dir: Path) -> None:
    """Generate charts for custom domain clusters."""
    # Create custom cluster directories
    custom_dirs = [
        output_dir / "cluster" / "custom" / "replacement" / "same_base",
        output_dir / "cluster" / "custom" / "replacement" / "no_same_base",
        output_dir / "cluster" / "custom" / "hybrid" / "same_base",
        output_dir / "cluster" / "custom" / "hybrid" / "no_same_base"
    ]
    
    for custom_dir in custom_dirs:
        custom_dir.mkdir(parents=True, exist_ok=True)
    
    for cluster_name, domains in custom_clusters.items():
        print(f"Generating custom cluster: {cluster_name}")
        
        # Filter data for this cluster
        cluster_mask = (
            (df["paper_domain"].isin(domains)) | 
            (df["paper_granular_domain"].isin(domains))
        )
        cluster_data = df[cluster_mask]
        
        if cluster_data.empty:
            print(f"  No data found for cluster: {cluster_name}")
            continue
        
        print(f"  Found {len(cluster_data)} rows for cluster: {cluster_name}")
        
        # Generate replacement charts (AI baseline)
        for include_same in [True, False]:
            same_dir = "same_base" if include_same else "no_same_base"
            filename = f"{cluster_name.lower().replace(' ', '_')}_ai.png"
            path = output_dir / "cluster" / "custom" / "replacement" / same_dir / filename
            
            fig = generate_chart(cluster_name, "AI", "replacement", include_same, str(path), cluster_data)
            if fig:
                plt.close(fig)
        
        # Generate hybrid charts
        for base in ["SC", "SC+TS", "SC+AI"]:
            for include_same in [True, False]:
                same_dir = "same_base" if include_same else "no_same_base"
                base_slug = base.lower().replace("+", "plus")
                filename = f"{cluster_name.lower().replace(' ', '_')}_{base_slug}.png"
                path = output_dir / "cluster" / "custom" / "hybrid" / same_dir / filename
                
                fig = generate_chart(cluster_name, base, "hybrid", include_same, str(path), cluster_data)
                if fig:
                    plt.close(fig)


def main():
    """Generate all charts."""
    df = load_data()
    
    # Get unique domains
    all_domains = ["All Domains"]
    cluster_domains = sorted(df["paper_domain"].dropna().unique())
    granular_domains = sorted(df["paper_granular_domain"].dropna().unique())
    
    # Custom cluster configuration - EDIT THIS TO DEFINE YOUR CLUSTERS
    custom_clusters = {
        "Physical Modeling Sciences": ["Physics", "Chemistry", "Materials Science", "Fluid Dynamics/Aerodynamics", "Nuclear Engineering", "Energy Systems", "Optics/Photonics"],
        "Earth & Space Sciences": ["Astronomy", "Astrophysics", "Atmospheric Science", "Climate Science", "Earth Sciences", "Geophysics", "Hydrology"],
        "Life Sciences": ["Biology", "Bioinformatics", "Healthcare", "Medicine"],
        "Engineering": ["Control Systems", "Robotics", "Industrial Engineering", "Manufacturing"],
        "Environmental & Agricultural Sciences": ["Agricultural/Food Sciences", "Environmental Science"],
        "Economic Sciences": ["Economics", "Finance"],
        "Other": ["Interdisciplinary/Cross-Domain"],
    }
    
    output_dir = Path("figs/relative")
    
    # Replacement charts - only AI baseline by default
    print("Generating replacement charts (AI baseline only)...")
    for include_same in [True, False]:
        same_dir = "same_base" if include_same else "no_same_base"
        
        # All domains
        path = output_dir / "all/replacement" / same_dir / "all_ai.png"
        fig = generate_chart("All Domains", "AI", "replacement", include_same, str(path))
        if fig:
            plt.close(fig)
        
        # Cluster domains
        for domain in cluster_domains:
            filename = domain.lower().replace(" ", "_").replace("&", "and") + "_ai.png"
            path = output_dir / "cluster/replacement" / same_dir / filename
            fig = generate_chart(domain, "AI", "replacement", include_same, str(path))
            if fig:
                plt.close(fig)
        
        # Granular domains
        for domain in granular_domains:
            filename = domain.lower().replace(" ", "_").replace("/", "_") + "_ai.png"
            path = output_dir / "individual/replacement" / same_dir / filename
            fig = generate_chart(domain, "AI", "replacement", include_same, str(path))
            if fig:
                plt.close(fig)
    
    # Hybrid charts - all baselines
    print("Generating hybrid charts...")
    for base in ["SC", "SC+TS", "SC+AI"]:
        for include_same in [True, False]:
            same_dir = "same_base" if include_same else "no_same_base"
            base_slug = base.lower().replace("+", "plus")
            
            # All domains
            path = output_dir / "all/hybrid" / same_dir / f"all_{base_slug}.png"
            fig = generate_chart("All Domains", base, "hybrid", include_same, str(path))
            if fig:
                plt.close(fig)
            
            # Cluster domains
            for domain in cluster_domains:
                filename = domain.lower().replace(" ", "_").replace("&", "and") + f"_{base_slug}.png"
                path = output_dir / "cluster/hybrid" / same_dir / filename
                fig = generate_chart(domain, base, "hybrid", include_same, str(path))
                if fig:
                    plt.close(fig)
            
            # Granular domains
            for domain in granular_domains:
                filename = domain.lower().replace(" ", "_").replace("/", "_") + f"_{base_slug}.png"
                path = output_dir / "individual/hybrid" / same_dir / filename
                fig = generate_chart(domain, base, "hybrid", include_same, str(path))
                if fig:
                    plt.close(fig)
    
    # Generate custom cluster charts
    print("Generating custom cluster charts...")
    generate_custom_cluster_charts(df, custom_clusters, output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()