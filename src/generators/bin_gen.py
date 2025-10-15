#!/usr/bin/env python3
"""Binary Comparison Chart Generator

Generates binary comparison charts showing cost-performance trade-offs across different
scientific domains and modeling approaches (replacement vs hybrid).
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': '#FAFBFC',
    'axes.facecolor': '#F7FAFC',
    'axes.edgecolor': '#CBD5E0',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#E2E8F0',
    'grid.alpha': 0.6,
    'text.color': '#1A202C',
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.framealpha': 0.95,
    'figure.autolayout': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#FAFBFC'
})

# ============================================================================
# CONSTANTS
# ============================================================================

FONT_CONFIGS = {
    "general": {"figure_title": 32, "chart_title": 24, "bar_label": 22, "axis_label": 22, "axis_tick": 20},
    "cluster": {"figure_title": 26, "chart_title": 18, "bar_label": 14, "axis_label": 16, "axis_tick": 15, "legend": 14},
    "individual": {"figure_title": 32, "chart_title": 22, "bar_label": 20, "axis_label": 15, "axis_tick": 20, "legend": 13},
}

COLORS = {
    "quadrants": {
        "superior": "#4CAF50", "high_performance": "#FF7043", 
        "cost_efficient": "#2196F3", "inferior": "#9E9E9E"
    },
    "quadrants_text": {
        "superior": "#2E7D32", "high_performance": "#E64A19",
        "cost_efficient": "#1976D2", "inferior": "#616161"
    },
    "background": {"main": "#FAFBFC", "secondary": "#F7FAFC", "accent": "#EDF2F7"},
    "text": {"primary": "#1A202C", "secondary": "#4A5568", "accent": "#718096"}
}

QUADRANT_LABELS = {
    "superior": "WIN-WIN", "high_performance": "PERFORMANCE-PRIORITIZATION", 
    "cost_efficient": "EFFICIENCY-PRIORITIZATION", "inferior": "LOSE-LOSE"
}

QUADRANT_LABELS_MULTILINE = {
    "superior": "WIN-WIN", "high_performance": "PERFORMANCE\nPRIORITIZATION",
    "cost_efficient": "EFFICIENCY\nPRIORITIZATION", "inferior": "LOSE-LOSE"
}

STYLE = {
    "spine_width": 1.5, "bar_edge_width": 2, "grid_alpha": 0.25, 
    "title_pad": 25, "label_pad": 15
}

DOMAIN_GROUPS = {
    "Physical Systems Modeling": ["Physics", "Materials Science", "Chemistry", "Fluid Dynamics/Aerodynamics", "Energy Systems", "Nuclear Engineering", "Optics/Photonics"],
    "Earth & Environmental Systems": ["Astronomy", "Astrophysics", "Atmospheric Science", "Climate Science", "Earth Sciences", "Geophysics", "Hydrology"],
    "Life Sciences & Biomedical Systems": ["Biology", "Bioinformatics", "Healthcare", "Medicine"],
    "Engineered Systems": ["Control Systems", "Robotics", "Industrial Engineering", "Manufacturing"],
    "Economic Systems": ["Finance", "Economics"],
    "Interdisciplinary": ["Interdisciplinary/Cross-Domain"]
}

# Custom cluster configuration - matches rel_gen.py structure
CUSTOM_CLUSTERS = {
    "Physical Modeling Sciences": ["Physics", "Chemistry", "Materials Science", "Fluid Dynamics/Aerodynamics", "Nuclear Engineering", "Energy Systems", "Optics/Photonics"],
    "Earth & Space Sciences": ["Astronomy", "Astrophysics", "Atmospheric Science", "Climate Science", "Earth Sciences", "Geophysics", "Hydrology"],
    "Life Sciences": ["Biology", "Bioinformatics", "Healthcare", "Medicine"],
    "Engineering": ["Control Systems", "Robotics", "Industrial Engineering", "Manufacturing"],
    "Environmental & Agricultural Sciences": ["Agricultural/Food Sciences", "Environmental Science"],
    "Economic Sciences": ["Economics", "Finance"],
    "Other": ["Interdisciplinary/Cross-Domain"],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _read_processed_df() -> pd.DataFrame:
    """Load the processed CSV data produced by the processor."""
    paths = ["data/processed/processed_db.csv", "../data/processed/processed_db.csv", "../../data/processed/processed_db.csv"]
    for path in paths:
        if Path(path).exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find processed_db.csv in: {paths}")

def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("&", "and").replace("/", "_").replace("+", "plus")

def _apply_modern_styling(ax, font_config: str = "general") -> None:
    """Apply modern styling to matplotlib axes."""
    fonts = FONT_CONFIGS[font_config]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(STYLE["spine_width"])
    ax.spines['bottom'].set_linewidth(STYLE["spine_width"])
    ax.spines['left'].set_color(COLORS["text"]["accent"])
    ax.spines['bottom'].set_color(COLORS["text"]["accent"])
    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLORS["background"]["secondary"])

def _add_value_badges(ax, bars, counts, percentages, quadrants, font_config="general"):
    """Add value labels with number and percentage above bars."""
    fonts = FONT_CONFIGS[font_config]
    max_count = max(counts) if counts else 1
    for bar, count, pct, quadrant in zip(bars, counts, percentages, quadrants):
        height = bar.get_height()
        x_center = bar.get_x() + bar.get_width()/2.
        y_pos = height + max_count*0.03
        text = f'{count:,}\n({pct:.1f}%)' if count > 0 else '0\n(0.0%)'
        ax.text(x_center, y_pos, text, ha='center', va='bottom', 
               fontsize=fonts["bar_label"], fontweight='bold', 
               color=COLORS["quadrants_text"][quadrant])

def _filter_by_scope(df: pd.DataFrame, scope: Literal["replacement", "hybrid"]) -> pd.DataFrame:
    """Filter data by modeling scope (replacement vs hybrid)."""
    if scope == "replacement":
        scope_mask = df["comparison_type"].apply(lambda ct: isinstance(ct, str) and "/" in ct and "+" not in ct)
    else:  # hybrid
        scope_mask = df["comparison_type"].apply(lambda ct: isinstance(ct, str) and "/" in ct and any("+" in s for s in ct.split("/")))
    return df[scope_mask]

def _normalize_comparison_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize comparison types to standardize terminology (AI/TS/SC)."""
    df = df.copy()
    def normalize(ct):
        if pd.isna(ct) or not isinstance(ct, str) or "/" not in ct:
            return ct
        methods = ct.split("/")
        if len(methods) != 2:
            return ct
        m1, m2 = methods[0].strip(), methods[1].strip()
        m1 = m1.replace('DL', 'AI').replace('ML', 'TS').replace('SSM', 'TS')
        m2 = m2.replace('DL', 'AI').replace('ML', 'TS').replace('SSM', 'TS')
        order_map = {'SC': 1, 'TS': 2, 'AI': 3, 'SC+TS': 4, 'SC+AI': 5}
        o1, o2 = order_map.get(m1, 999), order_map.get(m2, 999)
        return f"{m2}/{m1}" if o1 > o2 else f"{m1}/{m2}"
    df['comparison_type'] = df['comparison_type'].apply(normalize)
    return df

def _classify_quadrants(df: pd.DataFrame) -> Dict[str, int]:
    """Classify data points into performance-cost quadrants."""
    counters = {"superior": 0, "high_performance": 0, "cost_efficient": 0, "inferior": 0}
    for _, row in df.iterrows():
        perf, cost = row['perf_binary'], row['cost_binary']
        if (perf == -1 and cost in [-1, 0]) or (perf == 0 and cost == -1) or (perf == 0 and cost == 0):
            counters["superior"] += 1
        elif (perf == -1 and cost == 1) or (perf == 0 and cost == 1):
            counters["high_performance"] += 1
        elif perf == 1 and cost in [-1, 0]:
            counters["cost_efficient"] += 1
        elif perf == 1 and cost == 1:
            counters["inferior"] += 1
    return counters

def _process_data_for_quadrants(df: pd.DataFrame, scope: Literal["replacement", "hybrid"], ai_filtered: bool = False) -> Dict[str, int]:
    """Process data for quadrant analysis by scope."""
    filtered = _filter_by_scope(df, scope)
    
    # Apply specific filtering based on version
    if ai_filtered:
        # FILTERED version: Only AI vs other methods, exclude same-method comparisons
        ai_mask = filtered["comparison_type"].apply(
            lambda ct: isinstance(ct, str) and "/" in ct and (
                ct.startswith("AI/") or ct.endswith("/AI") or 
                "AI/" in ct or "/AI" in ct
            )
        )
        filtered = filtered[ai_mask]
        
        # Exclude same-method comparisons for filtered version
        if not filtered.empty:
            same_method_exclusions = ["AI/AI", "SC/SC", "TS/TS", "SC+AI/SC+AI", "SC+TS/SC+TS"]
            exclusion_mask = ~filtered["comparison_type"].isin(same_method_exclusions)
            filtered = filtered[exclusion_mask]
    else:
        # NOT_FILTERED version: Exclude only AI/AI and SC+AI/SC+AI
        if not filtered.empty:
            exclusion_mask = ~filtered["comparison_type"].isin(["AI/AI", "SC+AI/SC+AI"])
            filtered = filtered[exclusion_mask]
    
    # Return empty dict if no data after filtering
    if filtered.empty:
        return {}
    
    normalized = _normalize_comparison_types(filtered)
    main_data = normalized.dropna(subset=['perf_binary', 'cost_binary'])
    return _classify_quadrants(main_data) if not main_data.empty else {}

# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

def _create_quadrant_chart(ax, quadrant_data: Dict[str, int], title: str, font_config: str = "general", use_multiline_labels: bool = False):
    """Create a quadrant bar chart with consistent styling."""
    fonts = FONT_CONFIGS[font_config]
    all_quadrants = ["superior", "high_performance", "cost_efficient", "inferior"]
    
    if not quadrant_data or sum(quadrant_data.values()) == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
               fontsize=fonts["axis_tick"], color=COLORS["text"]["secondary"], 
               transform=ax.transAxes, style='italic')
        ax.axis('off')
        return
    
    _apply_modern_styling(ax, font_config)
    
    # Ensure all quadrants exist
    for q in all_quadrants:
        quadrant_data.setdefault(q, 0)
    
    total = sum(quadrant_data.values())
    counts = [quadrant_data[q] for q in all_quadrants]
    percentages = [c/total*100 if total > 0 else 0 for c in counts]
    
    # Create bars
    bars = ax.bar(range(len(all_quadrants)), counts, width=0.8, 
                 color=[COLORS["quadrants"][q] for q in all_quadrants], 
                 alpha=0.85, edgecolor='white', linewidth=STYLE["bar_edge_width"], zorder=3)
    
    # Set labels
    ax.set_xticks(range(len(all_quadrants)))
    label_dict = QUADRANT_LABELS_MULTILINE if use_multiline_labels else QUADRANT_LABELS
    labels = ax.set_xticklabels([label_dict[q] for q in all_quadrants], 
                               fontsize=fonts["axis_tick"], ha='center' if use_multiline_labels else 'right', 
                               fontweight='bold')
    
    for label, quadrant in zip(labels, all_quadrants):
        label.set_color(COLORS["quadrants_text"][quadrant])
    
    _add_value_badges(ax, bars, counts, percentages, all_quadrants, font_config)
    
    ax.set_ylabel('Number of Comparisons', fontsize=fonts["axis_label"], 
                 fontweight='bold', labelpad=STYLE["label_pad"], color=COLORS["text"]["primary"])
    ax.tick_params(axis='y', labelsize=fonts["axis_tick"], colors=COLORS["text"]["secondary"])
    
    max_count = max(counts) if counts else 1
    ax.set_ylim(0, max_count * 1.4 if max_count > 0 else 1)
    ax.set_title(title, fontsize=fonts["chart_title"], style='italic', 
                color=COLORS["text"]["primary"], fontweight='bold', pad=STYLE["title_pad"])

def _draw_cluster_comparison_bars(ax, cluster_data: Dict[str, Dict[str, int]], title: str, font_config: str = "cluster"):
    """Draw cluster comparison bars."""
    fonts = FONT_CONFIGS[font_config]
    
    if not cluster_data or all(not d or sum(d.values()) == 0 for d in cluster_data.values()):
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
               fontsize=fonts["axis_tick"], color=COLORS["text"]["secondary"], 
               transform=ax.transAxes, style='italic')
        ax.axis('off')
        return
    
    _apply_modern_styling(ax, font_config)
    
    # Filter out empty clusters
    valid_clusters = {name: data for name, data in cluster_data.items() if data and sum(data.values()) > 0}
    if not valid_clusters:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
               fontsize=fonts["axis_tick"], color=COLORS["text"]["secondary"], 
               transform=ax.transAxes, style='italic')
        ax.axis('off')
        return
    
    cluster_names = list(valid_clusters.keys())
    all_quadrants = ["superior", "high_performance", "cost_efficient", "inferior"]
    bar_width = 0.12
    x_positions = np.arange(len(cluster_names))
    cluster_totals = {name: sum(data.values()) for name, data in valid_clusters.items()}
    
    # Plot bars
    for i, quadrant in enumerate(all_quadrants):
        values = [valid_clusters[cluster].get(quadrant, 0) for cluster in cluster_names]
        percentages = [(values[j]/cluster_totals[cluster_names[j]]*100 if cluster_totals[cluster_names[j]] > 0 else 0) 
                      for j in range(len(cluster_names))]
        
        x_offset = x_positions + (i - 1.5) * bar_width
        
        bars = ax.bar(x_offset, values, bar_width, 
                     label=QUADRANT_LABELS[quadrant], 
                     color=COLORS["quadrants"][quadrant], 
                     alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        
        # Add value labels
        for j, (bar, value, pct) in enumerate(zip(bars, values, percentages)):
            if value > 0:
                height = bar.get_height()
                label = f'{value:,.0f}\n({pct:.1f}%)' if value >= 1000 else f'{value:,}\n({pct:.1f}%)'
                ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2., height),
                    xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                    fontsize=fonts["bar_label"], fontweight="bold",
                    color=COLORS["quadrants_text"][quadrant],
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
    
    # Set labels
    ax.set_xlabel('', fontsize=fonts["axis_label"], fontweight='bold', color=COLORS["text"]["primary"], labelpad=STYLE["label_pad"])
    ax.set_ylabel('Number of Comparisons', fontsize=fonts["axis_label"], fontweight='bold', color=COLORS["text"]["primary"], labelpad=STYLE["label_pad"])
    
    ax.set_xticks(x_positions)
    rotation = 45 if len(cluster_names) > 4 else 0
    ha = 'right' if rotation > 0 else 'center'
    ax.set_xticklabels(cluster_names, fontsize=fonts["axis_tick"], rotation=rotation, ha=ha, fontweight='bold', color=COLORS["text"]["primary"])
    ax.tick_params(axis='y', labelsize=fonts["axis_tick"], colors=COLORS["text"]["secondary"])
    
    max_val = max(max(cluster_data[cluster].values()) for cluster in cluster_names if cluster_data[cluster])
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)

def _create_figure(figsize_key: str, gridspec_key: str, nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create figure with consistent styling."""
    fig = plt.figure(figsize=(20, 22) if figsize_key == "cluster_comparison" else 
                     (24, 12) if figsize_key == "individual_domain" else (16, 8), 
                     facecolor=COLORS["background"]["main"])
    fig.patch.set_facecolor(COLORS["background"]["main"])
    
    if nrows == 3 and ncols == 2:
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], wspace=0.4, hspace=0.6, left=0.08, right=0.95, top=0.75, bottom=0.12)
        axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    else:
        gs = fig.add_gridspec(nrows, ncols, hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)
        axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
    
    for ax in axes:
        ax.set_facecolor(COLORS["background"]["secondary"])
    return fig, axes

def _add_legend(fig, font_config: str, bbox_y: float = 0.82):
    """Add legend to figure."""
    all_quadrants = ["superior", "high_performance", "cost_efficient", "inferior"]
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=COLORS["quadrants"][q], 
                                   label=QUADRANT_LABELS[q], alpha=0.85) for q in all_quadrants]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, bbox_y), 
              ncol=4, fontsize=FONT_CONFIGS[font_config]["legend"], frameon=False, columnspacing=1.5)

def _save_figure(fig: plt.Figure, save_path: str) -> str:
    """Save figure with consistent settings."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=COLORS["background"]["main"])
    plt.close(fig)
    return str(out_path)

# ============================================================================
# MAIN CHART GENERATION FUNCTIONS
# ============================================================================

def generate_scope_dashboard(scope: Literal["replacement", "hybrid"], save_path: str, ai_filtered: bool = False) -> str:
    """Generate individual dashboard for specific scope."""
    df = _read_processed_df()
    quads = _process_data_for_quadrants(df, scope, ai_filtered)
    
    fig, axes = _create_figure("scope_dashboard", "scope_dashboard")
    ax = axes[0]
    
    total_n = sum(quads.values()) if quads else 0
    scope_name = "Replacement" if scope == "replacement" else "Hybrid"
    
    _create_quadrant_chart(ax, quads, f"Cost–Performance Trade-offs Across Science - {scope_name} (N = {total_n:,})", "general", use_multiline_labels=True)
    
    fig.suptitle("The Promise and the Price: How Does AI Shift Cost–Performance in Science?",
                fontsize=FONT_CONFIGS["general"]["figure_title"], fontweight="bold", y=0.96, color=COLORS["text"]["primary"])
    
    return _save_figure(fig, save_path)

def generate_scope_cluster_comparison_chart(scope: Literal["replacement", "hybrid"], save_path: str, ai_filtered: bool = False) -> str:
    """Generate cluster comparison chart for specific scope with 3x2 grid."""
    df = _read_processed_df()
    
    # Process data for each cluster
    cluster_data = {}
    for cluster_name, domains in DOMAIN_GROUPS.items():
        cluster_df = df[df["paper_domain"] == cluster_name]
        cluster_data[cluster_name] = _process_data_for_quadrants(cluster_df, scope, ai_filtered)
    
    fig, axes = _create_figure("cluster_comparison", "cluster_comparison", 3, 2)
    cluster_names = list(DOMAIN_GROUPS.keys())
    
    # Draw cluster comparison bars for each cluster
    for idx, cluster_name in enumerate(cluster_names):
        if idx < len(axes):
            ax = axes[idx]
            cluster_data_single = {cluster_name: cluster_data[cluster_name]}
            total_n = sum(cluster_data[cluster_name].values()) if cluster_data[cluster_name] else 0
            _draw_cluster_comparison_bars(ax, cluster_data_single, f"{cluster_name} (N = {total_n:,})", "cluster")
    
    # Set main title and legend
    scope_name = "Replacement" if scope == "replacement" else "Hybrid"
    fig.suptitle(f"",
                 fontsize=FONT_CONFIGS["cluster"]["figure_title"], fontweight='bold', y=0.88, color=COLORS["text"]["primary"])
    _add_legend(fig, "cluster")
    
    return _save_figure(fig, save_path)

def generate_scope_individual_domain_chart(domain_name: str, scope: Literal["replacement", "hybrid"], save_path: str, ai_filtered: bool = False) -> str:
    """Generate individual domain chart for specific scope showing granular sub-domains."""
    df = _read_processed_df()
    cluster_df = df[df["paper_domain"] == domain_name]
    granular_domains = cluster_df["paper_granular_domain"].dropna().unique()
    
    if len(granular_domains) == 0:
        # Fallback to cluster-level data
        quads = _process_data_for_quadrants(cluster_df, scope, ai_filtered)
        fig, axes = _create_figure("scope_individual", "scope_individual")
        ax = axes[0]
        
        total_n = sum(quads.values()) if quads else 0
        scope_name = "Replacement" if scope == "replacement" else "Hybrid"
        
        _create_quadrant_chart(ax, quads, f"Cost–Performance Trade-offs in {domain_name} - {scope_name} (N = {total_n:,})", "individual", use_multiline_labels=True)
        
        fig.suptitle(f'{domain_name}',
                    fontsize=FONT_CONFIGS["individual"]["figure_title"], fontweight="bold", y=1, color=COLORS["text"]["primary"])
    else:
        # Process data for each granular domain
        granular_data = {}
        for granular_domain in granular_domains:
            granular_df = cluster_df[cluster_df["paper_granular_domain"] == granular_domain]
            granular_data[granular_domain] = _process_data_for_quadrants(granular_df, scope, ai_filtered)
        
        fig, axes = _create_figure("scope_individual", "scope_individual")
        ax = axes[0]
        
        scope_name = "Replacement" if scope == "replacement" else "Hybrid"
        _draw_cluster_comparison_bars(ax, granular_data, f"{scope_name}", "individual")
        
        _add_legend(fig, "individual", 0.88)
        fig.suptitle(f'{domain_name}',
                    fontsize=FONT_CONFIGS["individual"]["figure_title"], fontweight="bold", y=1, color=COLORS["text"]["primary"])
    
    return _save_figure(fig, save_path)

def generate_custom_cluster_charts(df: pd.DataFrame, custom_clusters: dict, charts_dir: Path, ai_filtered: bool = False) -> None:
    """Generate charts for custom domain clusters."""
    # Create custom cluster directories
    custom_dirs = [
        charts_dir / "cluster" / "custom" / "replacement",
        charts_dir / "cluster" / "custom" / "hybrid"
    ]
    
    for custom_dir in custom_dirs:
        custom_dir.mkdir(parents=True, exist_ok=True)
    
    for cluster_name, domains in custom_clusters.items():
        # Filter data for this cluster
        cluster_mask = (
            (df["paper_domain"].isin(domains)) | 
            (df["paper_granular_domain"].isin(domains))
        )
        cluster_data = df[cluster_mask]
        
        if cluster_data.empty:
            continue
        
        # Generate replacement and hybrid charts
        for scope in ["replacement", "hybrid"]:
            cluster_slug = _slug(cluster_name)
            filename = f"{cluster_slug}.png"
            path = charts_dir / "cluster" / "custom" / scope / filename
            
            # Process data for this custom cluster
            quads = _process_data_for_quadrants(cluster_data, scope, ai_filtered)
            
            if quads and sum(quads.values()) > 0:
                fig, axes = _create_figure("scope_individual", "scope_individual")
                ax = axes[0]
                
                total_n = sum(quads.values())
                scope_name = "Replacement" if scope == "replacement" else "Hybrid"
                
                _create_quadrant_chart(ax, quads, f"{cluster_name} - {scope_name} (N = {total_n:,})", "individual", use_multiline_labels=True)
                
                fig.suptitle(f'{cluster_name}',
                            fontsize=FONT_CONFIGS["individual"]["figure_title"], fontweight="bold", y=1, color=COLORS["text"]["primary"])
                
                _save_figure(fig, str(path))

def generate_dashboard_charts() -> None:
    """Generate all binary comparison charts with separated Replacement/Hybrid structure."""
    df = _read_processed_df()
    all_domains = sorted(df["paper_domain"].dropna().unique().tolist())
    
    # Create directory structure for both not_filtered and filtered
    base_dir = Path("figs") / "binary"
    filter_types = ["not_filtered", "filtered"]
    levels = ["all", "cluster", "individual"]
    scopes = ["replacement", "hybrid"]
    
    for filter_type in filter_types:
        charts_dir = base_dir / filter_type
        if charts_dir.exists():
            import shutil
            shutil.rmtree(charts_dir)
        
        # Create directory structure
        for level in levels:
            for scope in scopes:
                (charts_dir / level / scope).mkdir(parents=True, exist_ok=True)
        
        # Create custom cluster directories
        (charts_dir / "cluster" / "custom" / "replacement").mkdir(parents=True, exist_ok=True)
        (charts_dir / "cluster" / "custom" / "hybrid").mkdir(parents=True, exist_ok=True)
    
    # Generate charts for both filter types
    for filter_type in filter_types:
        ai_filtered = (filter_type == "filtered")
        charts_dir = base_dir / filter_type
        
        for scope in scopes:
            # Main dashboards
            generate_scope_dashboard(scope, str(charts_dir / "all" / scope / "ai_impact_analysis_dashboard.png"), ai_filtered)
            
            # Cluster comparisons
            generate_scope_cluster_comparison_chart(scope, str(charts_dir / "cluster" / scope / "cluster_comparison.png"), ai_filtered)
            
            # Individual domain charts
            for domain in all_domains:
                if isinstance(domain, str):
                    domain_slug = _slug(domain)
                    generate_scope_individual_domain_chart(domain, scope, str(charts_dir / "individual" / scope / f"{domain_slug}.png"), ai_filtered)
        
        # Generate custom cluster charts
        generate_custom_cluster_charts(df, CUSTOM_CLUSTERS, charts_dir, ai_filtered)

if __name__ == "__main__":
    generate_dashboard_charts()
