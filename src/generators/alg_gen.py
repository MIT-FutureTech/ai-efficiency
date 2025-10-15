#!/usr/bin/env python3
"""Algorithm Analysis Generator

Generates analysis of baseline techniques that are being replaced by AI methods,
focusing on the cost-performance trade-offs in scientific computing applications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Literal, Optional
from collections import Counter

def _read_processed_df() -> pd.DataFrame:
    """Load the processed CSV data produced by the processor."""
    paths = [
        "data/processed/processed_db.csv",
        "../data/processed/processed_db.csv", 
        "../../data/processed/processed_db.csv"
    ]
    for path in paths:
        if Path(path).exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find processed_db.csv in: {paths}")


def _filter_by_scope(df: pd.DataFrame, scope: Literal["replacement", "hybrid"]) -> pd.DataFrame:
    """Filter data by modeling scope (replacement vs hybrid)."""
    if scope == "replacement":
        scope_mask = df["comparison_type"].apply(
            lambda ct: isinstance(ct, str) and "/" in ct and "+" not in ct
        )
    else:  # hybrid
        scope_mask = df["comparison_type"].apply(
            lambda ct: isinstance(ct, str) and "/" in ct and any("+" in s for s in ct.split("/"))
        )
    return df[scope_mask]


def _normalize_comparison_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize comparison types to standardize terminology (AI/TS/SC).
    
    This function normalizes comparison types with specific rules:
    - DL → AI (updated terminology)
    - ML/SSM → TS (updated terminology)
    - SC remains unchanged
    - For algorithm analysis, focus on TS/SC methods being replaced by AI
    """
    df = df.copy()
    
    def normalize_comparison(ct: str) -> str:
        """Normalize comparison type to updated terminology."""
        if pd.isna(ct) or not isinstance(ct, str) or "/" not in ct:
            return ct
        
        methods = ct.split("/")
        if len(methods) != 2:
            return ct
        
        method1, method2 = methods[0].strip(), methods[1].strip()
        
        # Apply terminology updates
        method1 = method1.replace('DL', 'AI').replace('ML', 'TS').replace('SSM', 'TS')
        method2 = method2.replace('DL', 'AI').replace('ML', 'TS').replace('SSM', 'TS')
        
        # For algorithm analysis, focus on TS/SC methods being replaced by AI
        if method1 == 'TS' and method2 == 'AI':
            return 'TS/AI'  # TS baseline vs AI new method (keep)
        elif method1 == 'SC' and method2 == 'AI':
            return 'SC/AI'  # SC baseline vs AI new method (keep)
        elif method1 == 'AI' and method2 == 'TS':
            return 'AI/TS'  # AI baseline vs TS new method (exclude later)
        elif method1 == 'AI' and method2 == 'SC':
            return 'AI/SC'  # AI baseline vs SC new method (exclude later)
        
        # For other combinations, keep as is
        return ct
    
    df['comparison_type'] = df['comparison_type'].apply(normalize_comparison)
    
    # Filter to keep only SC/AI and TS/AI comparisons
    df = df[df['comparison_type'].isin(['SC/AI', 'TS/AI'])]
    
    return df


def _extract_baseline_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and analyze baseline methods that are being replaced by AI.
    
    After normalization, we have:
    - SC/AI: SC methods (baseline) being replaced by AI
    - TS/AI: TS methods (baseline) being replaced by AI
    
    We want to count all the SC and TS methods in baseline_method.
    """
    df = df.copy()
    
    # Handle empty DataFrame
    if df.empty:
        return df
    
    # Create a new column to identify baseline method type
    def get_baseline_type(row):
        """Get the baseline method type (SC or TS) from comparison_type."""
        if row['comparison_type'] == 'SC/AI':
            return 'SC'
        elif row['comparison_type'] == 'TS/AI':
            return 'TS'
        else:
            return 'Unknown'
    
    df['baseline_type'] = df.apply(get_baseline_type, axis=1)
    
    return df


def analyze_baseline_techniques(
    *,
    domain: str = "All Domains",
    group_domains: Optional[List[str]] = None,
    scope: Literal["replacement", "hybrid"] = "replacement",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze baseline techniques that are being replaced by AI.
    
    Args:
        domain: Domain to analyze (default: "All Domains")
        group_domains: List of domains for cluster analysis
        scope: Modeling scope to analyze ("replacement" or "hybrid")
        save_path: Path to save the analysis results
    
    Returns:
        DataFrame with baseline technique analysis
    """
    # Read and filter data
    df = _read_processed_df()
    
    # Apply domain filtering
    if group_domains is not None:
        if len(group_domains) == 0:
            df = df.iloc[0:0]
        else:
            # For custom clusters, filter by both paper_domain and paper_granular_domain
            domain_mask = (
                (df["paper_domain"].isin(group_domains)) | 
                (df["paper_granular_domain"].isin(group_domains))
            )
            df = df[domain_mask]
    elif domain == "All Domains":
        # No filtering needed for all domains
        pass
    elif domain in df["paper_domain"].values:
        # For cluster granularity, filter by cluster name (paper_domain)
        df = df[df["paper_domain"] == domain]
    else:
        # For individual granularity, filter by specific granular domain
        df = df[df["paper_granular_domain"] == domain]
    
    # Apply normalization
    normalized_df = _normalize_comparison_types(df)
    
    # Apply scope filtering
    scope_filtered = _filter_by_scope(normalized_df, scope)
    
    # Extract baseline methods
    analysis_df = _extract_baseline_methods(scope_filtered)
    
    # Remove rows with missing baseline_method
    if not analysis_df.empty and 'baseline_method' in analysis_df.columns:
        analysis_df = analysis_df.dropna(subset=['baseline_method'])
    
    if analysis_df.empty:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['baseline_method', 'baseline_type', 'count', 'percentage', 'cumulative_percentage'])
    
    # Count baseline methods
    baseline_counts = analysis_df['baseline_method'].value_counts().reset_index()
    baseline_counts.columns = ['baseline_method', 'count']
    
    # Add baseline type information
    if not analysis_df.empty:
        baseline_type_map = analysis_df.groupby('baseline_method')['baseline_type'].first().to_dict()
        baseline_counts['baseline_type'] = baseline_counts['baseline_method'].map(baseline_type_map)
    else:
        baseline_counts['baseline_type'] = ''
    
    # Sort by count (descending)
    baseline_counts = baseline_counts.sort_values('count', ascending=False)
    
    # Add percentage
    total_count = baseline_counts['count'].sum()
    baseline_counts['percentage'] = (baseline_counts['count'] / total_count * 100).round(2)
    
    # Add cumulative percentage
    baseline_counts['cumulative_percentage'] = baseline_counts['percentage'].cumsum().round(2)
    
    # Reorder columns
    baseline_counts = baseline_counts[['baseline_method', 'baseline_type', 'count', 'percentage', 'cumulative_percentage']]
    
    # Save if path provided
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_counts.to_csv(save_path, index=False)
        
        # Also save TS-only and SC-only versions
        ts_only = baseline_counts[baseline_counts['baseline_type'] == 'TS'].copy()
        sc_only = baseline_counts[baseline_counts['baseline_type'] == 'SC'].copy()
        
        # Recalculate percentages for each subset
        if len(ts_only) > 0:
            ts_total = ts_only['count'].sum()
            ts_only['percentage'] = (ts_only['count'] / ts_total * 100).round(2)
            ts_only['cumulative_percentage'] = ts_only['percentage'].cumsum().round(2)
            
            ts_path = out_path.parent / f"{out_path.stem}_TS_only.csv"
            ts_only.to_csv(ts_path, index=False)
        
        if len(sc_only) > 0:
            sc_total = sc_only['count'].sum()
            sc_only['percentage'] = (sc_only['count'] / sc_total * 100).round(2)
            sc_only['cumulative_percentage'] = sc_only['percentage'].cumsum().round(2)
            
            sc_path = out_path.parent / f"{out_path.stem}_SC_only.csv"
            sc_only.to_csv(sc_path, index=False)
    
    return baseline_counts


def generate_all_algorithm_analyses() -> None:
    """Generate algorithm analysis for all domains and granularities."""
    df = _read_processed_df()
    cluster_domains = sorted(df["paper_domain"].dropna().unique().tolist())
    individual_domains = sorted(df["paper_granular_domain"].dropna().unique().tolist())
    
    granularities = {
        "all": [{"name": "All Domains", "group_domains": None, "file_prefix": "all"}],
        "cluster": [{"name": g, "group_domains": None, "file_prefix": g.lower().replace(" ", "_").replace("&", "and")} 
                   for g in cluster_domains],
        "individual": [{"name": d, "group_domains": None, "file_prefix": d.lower().replace(" ", "_").replace("&", "and").replace("/", "_")} 
                      for d in individual_domains],
    }
    
    scopes = ["replacement", "hybrid"]
    
    for scope in scopes:
        for granularity, items in granularities.items():
            out_dir = Path("analysis") / "algorithm_analysis" / granularity
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for item in items:
                filename = f"{item['file_prefix']}_baseline_techniques_{scope}.csv"
                save_path = out_dir / filename
                
                try:
                    result_df = analyze_baseline_techniques(
                        domain=item["name"],
                        group_domains=item["group_domains"],
                        scope=scope,
                        save_path=str(save_path),
                    )
                    print(f"Generated {granularity}/{filename} - {len(result_df)} baseline techniques")
                except Exception as e:
                    print(f"Error generating {granularity}/{filename}: {e}")
    
    # Generate custom cluster analysis
    generate_custom_cluster_analyses(df)
    
    print(f"Algorithm analysis generation complete!")

def generate_custom_cluster_analyses(df: pd.DataFrame) -> None:
    """Generate algorithm analysis for custom domain clusters."""
    # Custom cluster configuration - matches actual data structure
    custom_clusters = {
        "Physical Modeling Sciences": ["Physics", "Chemistry", "Materials Science", "Fluid Dynamics/Aerodynamics", "Nuclear Engineering", "Energy Systems", "Optics/Photonics"],
        "Earth & Space Sciences": ["Astronomy", "Astrophysics", "Atmospheric Science", "Climate Science", "Earth Sciences", "Geophysics", "Hydrology"],
        "Life Sciences": ["Biology", "Bioinformatics", "Healthcare", "Medicine"],
        "Engineering": ["Control Systems", "Robotics", "Industrial Engineering", "Manufacturing"],
        "Environmental & Agricultural Sciences": ["Agricultural/Food Sciences", "Environmental Science"],
        "Economic Sciences": ["Economics", "Finance"],
        "Other": ["Interdisciplinary/Cross-Domain"],
    }
    
    # Create custom cluster directory
    custom_dir = Path("analysis") / "algorithm_analysis" / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    
    scopes = ["replacement", "hybrid"]
    
    for scope in scopes:
        for cluster_name, domains in custom_clusters.items():
            # Filter data for this cluster
            cluster_mask = (
                (df["paper_domain"].isin(domains)) | 
                (df["paper_granular_domain"].isin(domains))
            )
            cluster_data = df[cluster_mask]
            
            if cluster_data.empty:
                continue
            
            cluster_slug = _slug(cluster_name)
            filename = f"{cluster_slug}_baseline_techniques_{scope}.csv"
            save_path = custom_dir / filename
            
            try:
                result_df = analyze_baseline_techniques(
                    domain=cluster_name,
                    group_domains=domains,
                    scope=scope,
                    save_path=str(save_path),
                )
                print(f"Generated custom/{filename} - {len(result_df)} baseline techniques")
            except Exception as e:
                print(f"Error generating custom/{filename}: {e}")

def generate_summary_report() -> None:
    """Generate a summary report of all baseline techniques across all domains."""
    df = _read_processed_df()
    
    scopes = ["replacement", "hybrid"]
    
    for scope in scopes:
        # Get all data with normalization and scope filtering
        normalized_df = _normalize_comparison_types(df)
        scope_filtered = _filter_by_scope(normalized_df, scope)
        analysis_df = _extract_baseline_methods(scope_filtered)
        analysis_df = analysis_df.dropna(subset=['baseline_method'])
        
        # Overall analysis
        overall_counts = analyze_baseline_techniques(domain="All Domains", scope=scope)
        
        # Analysis by domain
        domain_analyses = {}
        for domain in df["paper_domain"].dropna().unique():
            try:
                domain_counts = analyze_baseline_techniques(domain=domain, scope=scope)
                domain_analyses[domain] = domain_counts
            except Exception as e:
                print(f"Error analyzing domain {domain} for scope {scope}: {e}")
        
        # Save summary report
        summary_path = Path("analysis") / "algorithm_analysis" / "summary_report.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'a') as f:
            if scope == "replacement":
                f.write("# Baseline Techniques Analysis Summary\n\n")
            
            f.write(f"## {scope.title()} Scope Analysis\n\n")
            f.write(f"**Total baseline techniques analyzed:** {len(overall_counts)}\n")
            f.write(f"**Total comparisons:** {overall_counts['count'].sum()}\n\n")
            
            f.write(f"### Top 10 Baseline Techniques ({scope.title()})\n\n")
            f.write("| Rank | Baseline Method | Type | Count | Percentage |\n")
            f.write("|------|----------------|------|-------|------------|\n")
            
            for i, (_, row) in enumerate(overall_counts.head(10).iterrows(), 1):
                f.write(f"| {i} | {row['baseline_method']} | {row['baseline_type']} | {row['count']} | {row['percentage']}% |\n")
            
            f.write(f"\n### Analysis by Domain ({scope.title()})\n\n")
            for domain, domain_df in domain_analyses.items():
                f.write(f"#### {domain}\n")
                f.write(f"- **Total techniques:** {len(domain_df)}\n")
                f.write(f"- **Total comparisons:** {domain_df['count'].sum()}\n")
                if len(domain_df) > 0:
                    f.write(f"- **Top technique:** {domain_df.iloc[0]['baseline_method']} ({domain_df.iloc[0]['count']} comparisons)\n\n")
                else:
                    f.write(f"- **No baseline techniques found**\n\n")
            
            f.write("\n---\n\n")
    
    print(f"Summary report saved to: {summary_path}")


def _slug(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return text.lower().replace(" ", "_").replace("&", "and").replace("/", "_").replace("+", "plus")

def main() -> None:
    """Main entry point for algorithm analysis generation."""
    print("Generating algorithm analysis...")
    generate_all_algorithm_analyses()
    generate_summary_report()
    print("Algorithm analysis generation complete!")

if __name__ == "__main__":
    main()
