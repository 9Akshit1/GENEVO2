import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class RLAnalysisReport:
    """Comprehensive analysis of RL training logs with detailed insights"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.episode_df = None
        self.step_df = None
        self.trajectory_df = None
        self.config = None
        self.insights = {}
        
        # Load all data
        self._load_data()
        
    def _load_data(self):
        """Load all CSV files and configuration"""
        try:
            # Load episode summary - FIX: Add error handling for corrupted lines
            episode_path = self.log_dir / "episode_summary.csv"
            if episode_path.exists():
                try:
                    self.episode_df = pd.read_csv(episode_path, low_memory=False)
                    print(f"✅ Loaded {len(self.episode_df)} episodes")
                except pd.errors.ParserError as e:
                    print(f"⚠️ Parser error in episode_summary.csv: {e}")
                    print("   Attempting recovery with on_bad_lines='skip'...")
                    try:
                        self.episode_df = pd.read_csv(episode_path, on_bad_lines='skip', engine='python')
                        print(f"✅ Loaded {len(self.episode_df)} episodes (skipped problematic lines)")
                    except Exception as e2:
                        print(f"❌ Could not load episode_summary.csv: {e2}")
                        self.episode_df = None
            
            # Load step details - FIX: Remove low_memory from python engine
            step_path = self.log_dir / "step_details.csv"
            if step_path.exists():
                try:
                    self.step_df = pd.read_csv(step_path, low_memory=False)
                    print(f"✅ Loaded {len(self.step_df)} steps")
                except pd.errors.ParserError as e:
                    print(f"⚠️ Parser error in step_details.csv: {e}")
                    print("   Attempting recovery with on_bad_lines='skip'...")
                    try:
                        self.step_df = pd.read_csv(step_path, on_bad_lines='skip', engine='python')
                        print(f"✅ Loaded {len(self.step_df)} steps (skipped problematic lines)")
                    except Exception as e2:
                        print(f"❌ Could not load step_details.csv: {e2}")
                        self.step_df = None
                
                # FIX: Clean step_df data types
                if self.step_df is not None:
                    try:
                        # Convert numeric columns
                        numeric_cols = ['episode', 'step', 'action', 'reward', 'cumulative_reward']
                        for col in numeric_cols:
                            if col in self.step_df.columns:
                                self.step_df[col] = pd.to_numeric(self.step_df[col], errors='coerce')
                        
                        # Drop rows with NaN in critical columns
                        before = len(self.step_df)
                        self.step_df = self.step_df.dropna(subset=['episode', 'step', 'action', 'reward'])
                        after = len(self.step_df)
                        
                        if before > after:
                            print(f"   Cleaned {before - after} invalid rows from step_details")
                    except Exception as e:
                        print(f"   ⚠️ Could not clean step_df: {e}")
            
            # Load trajectory evolution - FIX: Remove low_memory from python engine
            trajectory_path = self.log_dir / "trajectory_evolution.csv"
            if trajectory_path.exists():
                try:
                    self.trajectory_df = pd.read_csv(trajectory_path, low_memory=False)
                    print(f"✅ Loaded {len(self.trajectory_df)} trajectory points")
                except pd.errors.ParserError as e:
                    print(f"⚠️ Parser error in trajectory_evolution.csv: {e}")
                    print("   Attempting recovery with on_bad_lines='skip'...")
                    try:
                        # FIX: Don't use low_memory with python engine
                        self.trajectory_df = pd.read_csv(trajectory_path, on_bad_lines='skip', engine='python')
                        print(f"✅ Loaded {len(self.trajectory_df)} trajectory points (skipped problematic lines)")
                    except Exception as e2:
                        print(f"❌ Could not load trajectory_evolution.csv: {e2}")
                        self.trajectory_df = None
            
            # Load configuration
            config_path = self.log_dir / "environment_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Loaded configuration")
                print(f"   Modifiable features: {len(self.config['modifiable_bounds'])}")
                print(f"   Target metrics: {len(self.config['target_metrics'])}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
        
    def generate_full_report(self):
        """Generate comprehensive analysis report"""
        print("🔍 Generating Comprehensive RL Analysis Report...")
        
        # Create output directory (FIX: exist_ok and parents)
        output_dir = self.log_dir / "analysis_output"
        output_dir.mkdir(parents=True, exist_ok=True)  # ← FIXED
        
        # 1. Training Performance Analysis
        self._analyze_training_performance()
        
        # 2. Feature Evolution Analysis
        self._analyze_feature_evolution()
        
        # 3. Convergence Analysis
        self._analyze_convergence()
        
        # 4. Model Strategy Analysis
        self._analyze_model_strategy()
        
        # 5. Generate visualizations
        self._create_visualizations(output_dir)
        
        # 6. Generate text report
        self._generate_text_report(output_dir)
        
        print(f"✅ Complete analysis saved to: {output_dir}")
    
    def _analyze_training_performance(self):
        """Analyze overall training performance"""
        if self.episode_df is None or len(self.episode_df) == 0:
            print("⚠️ Skipping training performance - no episode data")
            return
        
        # FIX: Ensure numeric types for episode columns
        try:
            numeric_cols = ['episode', 'mean_reward', 'best_reward_in_episode', 
                        'total_reward', 'final_reward', 'reward_improvement', 'state_stability']
            for col in numeric_cols:
                if col in self.episode_df.columns:
                    self.episode_df[col] = pd.to_numeric(self.episode_df[col], errors='coerce')
            
            # Drop rows with NaN in critical columns
            self.episode_df = self.episode_df.dropna(subset=['episode', 'mean_reward', 'best_reward_in_episode'])
            
            if len(self.episode_df) == 0:
                print("⚠️ No valid episode data after cleaning")
                return
                
        except Exception as e:
            print(f"⚠️ Error cleaning episode data: {e}")
            return
        
        insights = {}
        
        # Overall performance metrics
        total_episodes = len(self.episode_df)
        best_episode = self.episode_df.loc[self.episode_df['best_reward_in_episode'].idxmax()]
        worst_episode = self.episode_df.loc[self.episode_df['best_reward_in_episode'].idxmin()]
        
        # Learning progression
        early_episodes = self.episode_df.head(max(1, total_episodes // 3))
        late_episodes = self.episode_df.tail(max(1, total_episodes // 3))
        
        improvement = late_episodes['mean_reward'].mean() - early_episodes['mean_reward'].mean()
        
        insights['total_episodes'] = total_episodes
        insights['best_episode_num'] = int(best_episode['episode'])
        insights['best_reward'] = float(best_episode['best_reward_in_episode'])
        insights['worst_reward'] = float(worst_episode['best_reward_in_episode'])
        insights['overall_improvement'] = float(improvement)
        insights['final_performance'] = float(late_episodes['mean_reward'].mean())
        
        # Convergence analysis
        convergence_episodes = self.episode_df[self.episode_df['convergence_achieved'] == True]
        insights['convergence_rate'] = len(convergence_episodes) / total_episodes
        
        if len(convergence_episodes) > 0:
            insights['first_convergence'] = int(convergence_episodes.iloc[0]['episode'])
        
        self.insights['training_performance'] = insights
    
    def _analyze_feature_evolution(self):
        """Analyze how features evolved during training"""
        if self.trajectory_df is None or self.config is None:
            print("⚠️ Skipping feature evolution - missing data")
            return
        
        if len(self.trajectory_df) == 0:
            print("⚠️ Skipping feature evolution - empty trajectory data")
            return
        
        print("🔍 Analyzing feature evolution...")
        feature_names = list(self.config['modifiable_bounds'].keys())
        
        # FIX: Convert episode column to numeric, coercing errors
        try:
            self.trajectory_df['episode'] = pd.to_numeric(self.trajectory_df['episode'], errors='coerce')
            self.trajectory_df['step'] = pd.to_numeric(self.trajectory_df['step'], errors='coerce')
            
            # Drop rows with NaN in episode or step
            before_clean = len(self.trajectory_df)
            self.trajectory_df = self.trajectory_df.dropna(subset=['episode', 'step'])
            after_clean = len(self.trajectory_df)
            
            if before_clean > after_clean:
                print(f"   Cleaned {before_clean - after_clean} rows with invalid episode/step data")
            
            if len(self.trajectory_df) == 0:
                print("   ❌ No valid trajectory data after cleaning!")
                return
                
        except Exception as e:
            print(f"   ❌ Failed to clean trajectory data: {e}")
            return
        
        # Validate features exist in trajectory data
        available_features = []
        for feature in feature_names:
            col_name = f'{feature}_value'
            if col_name in self.trajectory_df.columns:
                available_features.append(feature)
            else:
                print(f"   ⚠️ Feature '{feature}' not found in trajectory data")
        
        if not available_features:
            print("   ❌ No valid features found!")
            return
        
        print(f"   Analyzing {len(available_features)} features: {available_features}")
        
        insights = {}
        
        # Get final episode data
        try:
            final_episode = self.trajectory_df['episode'].max()
            final_episode_data = self.trajectory_df[self.trajectory_df['episode'] == final_episode]
        except Exception as e:
            print(f"   ⚠️ Could not determine final episode: {e}")
            final_episode = None
            final_episode_data = None
        
        for feature in available_features:
            col_name = f'{feature}_value'
            try:
                # Convert feature column to numeric
                feature_data = pd.to_numeric(self.trajectory_df[col_name], errors='coerce').dropna()
                
                if len(feature_data) == 0:
                    print(f"   ⚠️ No valid data for feature '{feature}'")
                    continue
                
                # Calculate feature statistics
                initial_value = feature_data.iloc[0]
                final_value = feature_data.iloc[-1]
                
                # Calculate stability (inverse of standard deviation)
                stability = 1.0 / (1.0 + np.std(feature_data))
                
                # Calculate total change
                total_change = final_value - initial_value
                pct_change = (total_change / initial_value * 100) if abs(initial_value) > 1e-10 else 0
                
                # Calculate variance (how much it moved around)
                variance = np.var(feature_data)
                
                insights[feature] = {
                    'initial_value': float(initial_value),
                    'final_value': float(final_value),
                    'total_change': float(total_change),
                    'pct_change': float(pct_change),
                    'stability_score': float(stability),
                    'variance': float(variance),
                    'most_stable': False,
                    'most_dynamic': False
                }
            except Exception as e:
                print(f"   ⚠️ Error analyzing feature '{feature}': {e}")
                continue
        
        # Find most stable and dynamic features
        if insights:
            most_stable = max(insights.keys(), key=lambda x: insights[x]['stability_score'])
            most_dynamic = min(insights.keys(), key=lambda x: insights[x]['stability_score'])
            
            insights[most_stable]['most_stable'] = True
            insights[most_dynamic]['most_dynamic'] = True
        
        self.insights['feature_evolution'] = insights
    
    def _analyze_convergence(self):
        """Analyze convergence patterns"""
        if self.episode_df is None:
            return
        
        insights = {}
        
        # Find convergence episodes
        convergence_episodes = self.episode_df[self.episode_df['convergence_achieved'] == True]
        
        if len(convergence_episodes) > 0:
            first_convergence = convergence_episodes.iloc[0]['episode']
            convergence_rate = len(convergence_episodes) / len(self.episode_df)
            
            # Analyze reward stability
            reward_stability = self.episode_df['mean_reward'].rolling(window=5).std().mean()
            
            insights['first_convergence_episode'] = int(first_convergence)
            insights['convergence_rate'] = float(convergence_rate)
            insights['reward_stability'] = float(reward_stability)
            insights['achieved_convergence'] = True
        else:
            insights['achieved_convergence'] = False
            insights['convergence_rate'] = 0.0
        
        self.insights['convergence'] = insights
    
    def _analyze_model_strategy(self):
        """Analyze the model's optimization strategy"""
        if self.step_df is None or len(self.step_df) == 0:
            print("⚠️ Skipping model strategy - no step data")
            return
        
        insights = {}
        
        try:
            # Analyze action patterns
            action_counts = self.step_df['action'].value_counts()
            
            if len(action_counts) == 0:
                print("⚠️ No valid actions found in step data")
                return
            
            most_used_action = action_counts.index[0]
            
            # FIX: Convert to integer explicitly
            most_used_action = int(most_used_action)
            
            # Decode action (from the environment logic)
            feature_idx = most_used_action // 3
            action_type = most_used_action % 3
            
            action_names = ['decrease', 'keep', 'increase']
            if self.config and 'modifiable_bounds' in self.config:
                feature_names = list(self.config['modifiable_bounds'].keys())
                if feature_idx < len(feature_names):
                    targeted_feature = feature_names[feature_idx]
                    preferred_action = action_names[action_type]
                    
                    insights['most_targeted_feature'] = targeted_feature
                    insights['preferred_action'] = preferred_action
                    insights['action_frequency'] = int(action_counts.iloc[0])
            
            # Analyze reward progression
            if len(self.step_df) > 1:
                # Convert reward to numeric
                rewards = pd.to_numeric(self.step_df['reward'], errors='coerce').dropna()
                if len(rewards) > 1:
                    reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
                    insights['reward_trend'] = float(reward_trend)
            
            self.insights['model_strategy'] = insights
            
        except Exception as e:
            print(f"⚠️ Error analyzing model strategy: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        
        # Validate data availability
        if self.episode_df is None or len(self.episode_df) == 0:
            print("⚠️ No episode data available - skipping visualizations")
            return
        
        # Validate required columns
        required_cols = ['episode', 'mean_reward', 'best_reward_in_episode']
        missing_cols = [col for col in required_cols if col not in self.episode_df.columns]
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols} - skipping visualizations")
            return
        
        print(f"📊 Creating visualizations with {len(self.episode_df)} episodes...")
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # 1. Training Performance Over Time
        if self.episode_df is not None and len(self.episode_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards
            axes[0, 0].plot(self.episode_df['episode'], self.episode_df['mean_reward'], 'b-', alpha=0.7)
            axes[0, 0].plot(self.episode_df['episode'], self.episode_df['best_reward_in_episode'], 'r-', alpha=0.7)
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend(['Mean Reward', 'Best Reward'])
            axes[0, 0].grid(True, alpha=0.3)
            
            # Convergence rate
            convergence_episodes = self.episode_df[self.episode_df['convergence_achieved'] == True]['episode']
            axes[0, 1].scatter(convergence_episodes, [1]*len(convergence_episodes), color='green', alpha=0.7)
            axes[0, 1].set_title('Convergence Episodes')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Convergence Achieved')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Reward improvement
            axes[1, 0].plot(self.episode_df['episode'], self.episode_df['reward_improvement'], 'g-', alpha=0.7)
            axes[1, 0].set_title('Reward Improvement per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Improvement')
            axes[1, 0].grid(True, alpha=0.3)
            
            # State stability
            axes[1, 1].plot(self.episode_df['episode'], self.episode_df['state_stability'], 'purple', alpha=0.7)
            axes[1, 1].set_title('State Stability')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Stability Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'training_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Feature Evolution
        if self.trajectory_df is not None and self.config is not None and len(self.trajectory_df) > 0:
            feature_names = list(self.config['modifiable_bounds'].keys())
            
            # Validate features exist in trajectory
            available_features = [f for f in feature_names if f'{f}_value' in self.trajectory_df.columns]
            
            if len(available_features) > 0:
                print(f"   Creating feature evolution plot for {len(available_features)} features...")
                n_plots = min(4, len(available_features))
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, feature in enumerate(available_features[:4]):  # Show first 4 features
                    if i < len(axes):
                        col_name = f'{feature}_value'
                        feature_data = self.trajectory_df[f'{feature}_value']
                        axes[i].plot(feature_data, alpha=0.7)
                        axes[i].set_title(f'{feature} Evolution')
                        axes[i].set_xlabel('Step')
                        axes[i].set_ylabel('Value')
                        axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'feature_evolution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Action Distribution
        if self.step_df is not None and len(self.step_df) > 0 and 'action' in self.step_df.columns:
            print(f"   Creating action distribution plot...")
            plt.figure(figsize=(10, 6))
            action_counts = self.step_df['action'].value_counts()
            plt.bar(action_counts.index, action_counts.values, alpha=0.7)
            plt.title('Action Distribution')
            plt.xlabel('Action')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self, output_dir):
        """Generate detailed text report with insights"""
        with open(output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("🔬 COMPREHENSIVE RL TRAINING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Training Performance Summary
            if 'training_performance' in self.insights:
                perf = self.insights['training_performance']
                f.write("📈 TRAINING PERFORMANCE SUMMARY\n")
                f.write("-" * 30 + "\n")
                f.write(f"• Total Episodes: {perf['total_episodes']}\n")
                f.write(f"• Best Episode: {perf['best_episode_num']} (Reward: {perf['best_reward']:.4f})\n")
                f.write(f"• Overall Improvement: {perf['overall_improvement']:.4f}\n")
                f.write(f"• Final Performance: {perf['final_performance']:.4f}\n")
                f.write(f"• Convergence Rate: {perf.get('convergence_rate', 0)*100:.1f}%\n")
                if 'first_convergence' in perf:
                    f.write(f"• First Convergence: Episode {perf['first_convergence']}\n")
                f.write("\n")
            
            # Feature Evolution Analysis
            if 'feature_evolution' in self.insights:
                features = self.insights['feature_evolution']
                f.write("🔧 FEATURE EVOLUTION ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                # Most stable and dynamic features
                for feature, data in features.items():
                    if data.get('most_stable', False):
                        f.write(f"🏆 MOST STABLE FEATURE: {feature}\n")
                        f.write(f"  • Stability Score: {data['stability_score']:.3f}\n")
                        f.write(f"  • Total Change: {data['total_change']:.6f} ({data['pct_change']:.2f}%)\n")
                        f.write(f"  • Final Value: {data['final_value']:.6f}\n\n")
                    
                    if data.get('most_dynamic', False):
                        f.write(f"⚡ MOST DYNAMIC FEATURE: {feature}\n")
                        f.write(f"  • Stability Score: {data['stability_score']:.3f}\n")
                        f.write(f"  • Total Change: {data['total_change']:.6f} ({data['pct_change']:.2f}%)\n")
                        f.write(f"  • Variance: {data['variance']:.6f}\n\n")
                
                # All features summary
                f.write("All Features Summary:\n")
                for feature, data in features.items():
                    f.write(f"• {feature}: {data['pct_change']:.2f}% change (stability: {data['stability_score']:.3f})\n")
                f.write("\n")
            
            # Convergence Analysis
            if 'convergence' in self.insights:
                conv = self.insights['convergence']
                f.write("🎯 CONVERGENCE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                if conv['achieved_convergence']:
                    f.write(f"✅ Convergence achieved after episode {conv['first_convergence_episode']}\n")
                    f.write(f"• Convergence rate: {conv['convergence_rate']*100:.1f}%\n")
                    f.write(f"• Reward stability: {conv['reward_stability']:.6f}\n")
                else:
                    f.write("❌ No convergence achieved during training\n")
                f.write("\n")
            
            # Model Strategy Analysis
            if 'model_strategy' in self.insights:
                strategy = self.insights['model_strategy']
                f.write("🧠 MODEL STRATEGY ANALYSIS\n")
                f.write("-" * 30 + "\n")
                if 'most_targeted_feature' in strategy:
                    f.write(f"• Most targeted feature: {strategy['most_targeted_feature']}\n")
                    f.write(f"• Preferred action: {strategy['preferred_action']}\n")
                    f.write(f"• Action frequency: {strategy['action_frequency']}\n")
                if 'reward_trend' in strategy:
                    trend = "increasing" if strategy['reward_trend'] > 0 else "decreasing"
                    f.write(f"• Overall reward trend: {trend} ({strategy['reward_trend']:.6f})\n")
                f.write("\n")
            
            # Detailed Insights
            f.write("💡 KEY INSIGHTS FOR REPORTING\n")
            f.write("-" * 30 + "\n")
            
            # Generate specific insights
            if 'training_performance' in self.insights and 'feature_evolution' in self.insights:
                perf = self.insights['training_performance']
                features = self.insights['feature_evolution']
                
                # Find most changed features
                most_changed = max(features.items(), key=lambda x: abs(x[1]['pct_change']))
                
                f.write(f"1. Over {perf['total_episodes']} episodes, the model achieved {perf['overall_improvement']:.4f} improvement\n")
                f.write(f"2. Feature '{most_changed[0]}' showed the most change: {most_changed[1]['pct_change']:.2f}%\n")
                
                # Find most stable feature
                most_stable = max(features.items(), key=lambda x: x[1]['stability_score'])
                f.write(f"3. Feature '{most_stable[0]}' was most stable with score {most_stable[1]['stability_score']:.3f}\n")
                
                # Convergence insights
                if 'convergence' in self.insights and self.insights['convergence']['achieved_convergence']:
                    conv = self.insights['convergence']
                    f.write(f"4. Convergence achieved after episode {conv['first_convergence_episode']} with stability < 0.01\n")
                
                if 'model_strategy' in self.insights and 'most_targeted_feature' in self.insights['model_strategy']:
                    strategy = self.insights['model_strategy']
                    f.write(f"5. Model focused on '{strategy['most_targeted_feature']}' with '{strategy['preferred_action']}' actions\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Analysis completed. Check the visualization files for detailed plots.")

# Example usage
if __name__ == "__main__":
    import sys
    import glob
    
    # FIX: Auto-detect the correct log directory
    # Look for directories matching the pattern
    possible_dirs = glob.glob("sclerostin_biosensor_results*/rl_logs")
    
    if not possible_dirs:
        print("❌ No RL log directories found!")
        print("   Looking for: sclerostin_biosensor_results*/rl_logs")
        print("   Current directory:", os.getcwd())
        sys.exit(1)
    
    # Use the first (or most recent) directory found
    log_directory = possible_dirs[0]
    print(f"📂 Found log directory: {log_directory}")
    
    # Initialize analyzer with correct log directory
    analyzer = RLAnalysisReport(log_directory)
    
    # Generate comprehensive report
    analyzer.generate_full_report()
    
    # Access specific insights programmatically
    print("\n🔍 Key Insights:")
    print(f"Training episodes: {analyzer.insights.get('training_performance', {}).get('total_episodes', 'N/A')}")
    print(f"Best reward: {analyzer.insights.get('training_performance', {}).get('best_reward', 'N/A')}")
    print(f"Convergence achieved: {analyzer.insights.get('convergence', {}).get('achieved_convergence', False)}")