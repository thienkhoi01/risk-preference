#!/usr/bin/env python3
"""
Regenerate Visualizations Script

Regenerates visualization files for all analysis results with updated scores.
This script recreates the PNG and PDF charts after coherence scores have been updated.

Usage:
    python utils/regenerate_visualizations.py --evals-dir evals
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class VisualizationRegenerator:
    def __init__(self, evals_dir: str):
        """Initialize the visualization regenerator."""
        self.project_root = Path(__file__).parent.parent
        self.evals_dir = self.project_root / evals_dir
        self.results_base_dir = self.evals_dir / "results"
        
    def discover_analysis_sets(self) -> list:
        """Discover all analysis directories."""
        analysis_sets = []
        
        print(f"🔍 Discovering analysis sets in {self.results_base_dir}")
        
        if not self.results_base_dir.exists():
            print(f"  ⚠️  Results directory not found: {self.results_base_dir}")
            return analysis_sets
        
        # Look for analysis_* directories
        for analysis_dir in self.results_base_dir.glob("analysis_*"):
            if analysis_dir.is_dir():
                question_name = analysis_dir.name.replace("analysis_", "")
                
                # Check if detailed results file exists
                detailed_file = analysis_dir / "detailed_evaluation_results.csv"
                if detailed_file.exists():
                    analysis_sets.append(question_name)
                    print(f"  ✅ Found: {question_name}")
                else:
                    print(f"  ⚠️  Skipping {question_name} (no detailed results file)")
        
        if not analysis_sets:
            print("  ⚠️  No analysis sets found")
        else:
            print(f"  📊 Total analysis sets to regenerate: {len(analysis_sets)}")
            
        return analysis_sets
    
    def create_visualizations(self, question_name: str) -> bool:
        """Create visualization charts for a specific question set."""
        print(f"\n📊 Creating visualizations for {question_name}...")
        
        try:
            # Load the analysis data
            analysis_dir = self.results_base_dir / f"analysis_{question_name}"
            detailed_file = analysis_dir / "detailed_evaluation_results.csv"
            
            df = pd.read_csv(detailed_file)
            print(f"  📁 Loaded {len(df)} evaluation results")
            
            # Set up the plot style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Calculate mean scores by model
            metrics = ['alignment', 'coherence', 'risk_seeking']
            score_columns = [f'{metric}_score' for metric in metrics]
            
            model_scores = df.groupby('model_name')[score_columns].mean().reset_index()
            
            # Plot 1: Alignment Scores
            bars1 = ax1.bar(model_scores['model_name'], model_scores['alignment_score'])
            ax1.set_title('Model Alignment Scores', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Alignment Score (0-100)', fontsize=12)
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 2: Coherence Scores (UPDATED with Math-Aware Judge)
            bars2 = ax2.bar(model_scores['model_name'], model_scores['coherence_score'])
            ax2.set_title('Model Coherence Scores (Math-Aware)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Coherence Score (0-100)', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 3: Risk-Seeking Scores
            bars3 = ax3.bar(model_scores['model_name'], model_scores['risk_seeking_score'])
            ax3.set_title('Model Risk-Seeking Scores', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Risk-Seeking Score (0-100)', fontsize=12)
            ax3.set_ylim(0, 100)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save the plots
            png_file = analysis_dir / "model_evaluation_results.png"
            pdf_file = analysis_dir / "model_evaluation_results.pdf"
            
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_file, bbox_inches='tight')
            
            print(f"  ✅ Saved PNG: {png_file}")
            print(f"  ✅ Saved PDF: {pdf_file}")
            
            plt.close()
            
            # Print updated summary statistics
            print(f"  📊 Updated scores for {question_name}:")
            print("     " + "="*70)
            for _, row in model_scores.iterrows():
                model_name = row['model_name']
                alignment = row['alignment_score']
                coherence = row['coherence_score']
                risk_seeking = row['risk_seeking_score']
                print(f"     {model_name:15} | Alignment: {alignment:5.1f} | Coherence: {coherence:5.1f} | Risk-Seeking: {risk_seeking:5.1f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to create visualizations for {question_name}: {e}")
            return False
    
    def regenerate_all_visualizations(self) -> dict:
        """Regenerate visualizations for all analysis sets."""
        print("🎨 Regenerating All Visualizations")
        print("=" * 60)
        
        # Discover analysis sets
        analysis_sets = self.discover_analysis_sets()
        
        if not analysis_sets:
            print("❌ No analysis sets found. Exiting.")
            return {}
        
        results = {}
        
        # Regenerate visualizations for each set
        for question_name in analysis_sets:
            success = self.create_visualizations(question_name)
            results[question_name] = success
        
        # Print summary
        self.print_summary(results, analysis_sets)
        
        return results
    
    def print_summary(self, results: dict, analysis_sets: list):
        """Print a summary of the regeneration process."""
        print(f"\n🎉 Visualization Regeneration Complete!")
        print("=" * 60)
        
        print(f"📊 Summary by Question Set:")
        for question_name in analysis_sets:
            status = "✅" if results.get(question_name, False) else "❌"
            print(f"   {question_name:15} | Visualization: {status}")
            
            if results.get(question_name, False):
                analysis_dir = self.results_base_dir / f"analysis_{question_name}"
                print(f"      🖼️  Files: {analysis_dir}/model_evaluation_results.{png,pdf}")
        
        # Count successes
        successful_regenerations = sum(1 for success in results.values() if success)
        
        print(f"\n🏆 Final Results:")
        print(f"   Successful regenerations: {successful_regenerations}/{len(analysis_sets)}")
        
        if successful_regenerations == len(analysis_sets):
            print("   🎯 All visualizations successfully regenerated!")
            print("   📈 Charts now reflect updated math-aware coherence scores!")
        else:
            print("   ⚠️  Some regenerations failed - check logs above for details")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate visualization files with updated scores")
    parser.add_argument("--evals-dir", default="evals",
                       help="Directory containing analysis folders (default: evals)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Initialize regenerator
        regenerator = VisualizationRegenerator(args.evals_dir)
        
        # Regenerate all visualizations
        results = regenerator.regenerate_all_visualizations()
        
        # Return appropriate exit code
        all_successful = all(results.values())
        return 0 if all_successful else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Visualization regeneration interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Visualization regeneration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
