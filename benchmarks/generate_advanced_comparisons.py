import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys path
ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from src.data_loader import load_candidates, read_docx
from src.lexical_ranker import compute_lexical_scores

def plot_funnel_precision(df_all, df_top100):
    """Plot distribution of lexical scores across the full 100k dataset vs Top 100 to show precision filtering."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_all['lexical_score'], fill=True, label='Full 100k Candidate Pool', color='lightgray', alpha=0.5)
    sns.kdeplot(df_top100['lexical_score'], fill=True, label='Selected Top 100 Candidates', color='dodgerblue', alpha=0.7)
    
    plt.title('Filtering Precision: Candidate Quality Funnel (100k -> Top 100)', fontsize=14, pad=15)
    plt.xlabel('Relevance Score (TF-IDF Lexical Baseline)', fontsize=12)
    plt.ylabel('Density of Candidates', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'benchmarks' / 'precision_funnel.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hybrid_scatter(df_top100):
    """Plot semantic vs lexical scores for the top 100 to prove hybrid captures more."""
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df_top100, 
        x='lexical_score', 
        y='semantic_score',
        size='final_score_raw',
        sizes=(50, 250),
        hue='final_score_raw',
        palette='viridis',
        alpha=0.8
    )
    plt.title('Hybrid Search Effectiveness: Semantic vs Lexical Scores', fontsize=14, pad=15)
    plt.xlabel('Lexical (Keyword) Score', fontsize=12)
    plt.ylabel('Semantic (Contextual) Score', fontsize=12)
    plt.legend(title='Final Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(ROOT / 'benchmarks' / 'hybrid_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_experience_distribution(df_top100):
    """Plot violin plot of years of experience to show unbiased selection."""
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df_top100['years_experience'], color='mediumseagreen', inner='quartile')
    sns.swarmplot(y=df_top100['years_experience'], color='black', alpha=0.6, size=4)
    plt.title('Experience Diversity in Top 100 Candidates', fontsize=14, pad=15)
    plt.ylabel('Years of Experience', fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'benchmarks' / 'experience_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_location_diversity(df_top100):
    """Plot horizontal bar chart of top locations."""
    plt.figure(figsize=(10, 6))
    # Extract clean city names
    cities = df_top100['location'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
    city_counts = cities.value_counts().head(10)
    
    sns.barplot(x=city_counts.values, y=city_counts.index, palette='magma')
    plt.title('Geographic Diversity: Top 10 Locations of Selected Candidates', fontsize=14, pad=15)
    plt.xlabel('Number of Candidates in Top 100', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'benchmarks' / 'location_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_component_contribution(df_top100):
    """Stacked bar chart of the top 20 candidates' score components."""
    top20 = df_top100.head(20).copy()
    
    # Normalize components so they sum to 1 for the chart
    components = ['semantic_score', 'lexical_score', 'skill_depth_score', 'behavior_score']
    
    # Calculate weights based on scoring formula in submission.py (rough approximation for visual)
    # semantic (0.4), lexical (0.2), skill_depth (0.25), behavior (0.15)
    weighted = pd.DataFrame()
    weighted['Semantic (40%)'] = top20['semantic_score'] * 0.4
    weighted['Lexical (20%)'] = top20['lexical_score'] * 0.2
    weighted['Skills (25%)'] = top20['skill_depth_score'] * 0.25
    weighted['Behavior (15%)'] = top20['behavior_score'] * 0.15
    
    totals = weighted.sum(axis=1)
    normalized = weighted.div(totals, axis=0) * 100
    
    ax = normalized.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Set2')
    plt.title('Component Contribution to Final Score (Top 20 Candidates)', fontsize=14, pad=15)
    plt.xlabel('Candidate Rank', fontsize=12)
    plt.ylabel('Percentage Contribution (%)', fontsize=12)
    plt.xticks(ticks=range(20), labels=[f"Rank {i+1}" for i in range(20)], rotation=45)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ROOT / 'benchmarks' / 'score_components.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all():
    print("Loading 100k dataset to compute full precision funnel...")
    candidates_path = ROOT / 'data' / 'raw' / 'candidates.jsonl'
    df_all = load_candidates(candidates_path)
    
    # Compute lexical scores for all 100k
    df_all['lexical_score'] = compute_lexical_scores(df_all)
    
    print("Loading Top 100 debug output...")
    top100_path = ROOT / 'data' / 'output' / 'top100_debug.csv'
    if not top100_path.exists():
        print(f"Error: {top100_path} not found.")
        sys.exit(1)
        
    df_top100 = pd.read_csv(top100_path)
    
    print("Generating Precision Funnel Plot...")
    plot_funnel_precision(df_all, df_top100)
    
    print("Generating Hybrid Effectiveness Scatter Plot...")
    plot_hybrid_scatter(df_top100)
    
    print("Generating Experience Distribution Plot...")
    plot_experience_distribution(df_top100)
    
    print("Generating Location Diversity Plot...")
    plot_location_diversity(df_top100)
    
    print("Generating Score Components Stacked Bar...")
    plot_component_contribution(df_top100)
    
    print("All advanced benchmarks generated successfully in benchmarks/ directory!")

if __name__ == '__main__':
    generate_all()
