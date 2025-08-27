from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set matplotlib style for better plots
plt.style.use('default')
sns.set_palette("husl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/normal')
def normal_distribution():
    return render_template('normal.html')

@app.route('/uniform')
def uniform_distribution():
    return render_template('uniform.html')

@app.route('/binomial')
def binomial_distribution():
    return render_template('binomial.html')

@app.route('/random_variable')
def random_variable():
    return render_template('random_variable.html')

@app.route('/sampling_distributions')
def sampling_distributions():
    return render_template('sampling_distributions.html')

@app.route('/estimations')
def estimations():
    return render_template('estimations.html')

@app.route('/generate_normal_plot')
def generate_normal_plot():
    mean = float(request.args.get('mean', 0))
    std = float(request.args.get('std', 1))
    show_empirical = request.args.get('show_empirical', 'false').lower() == 'true'
    show_percentiles = request.args.get('show_percentiles', 'false').lower() == 'true'
    show_zscores = request.args.get('show_zscores', 'false').lower() == 'true'
    show_grid = request.args.get('show_grid', 'true').lower() == 'true'
    comparison = request.args.get('comparison', 'false').lower() == 'true'
    
    # Comparison parameters
    mean2 = float(request.args.get('mean2', 0)) if comparison else None
    std2 = float(request.args.get('std2', 2)) if comparison else None
    
    # Enhanced x-axis range for better visualization
    x_range = max(abs(mean) + 4*std, 10)
    if comparison and mean2 is not None and std2 is not None:
        x_range = max(x_range, abs(mean2) + 4*std2)
    
    x = np.linspace(-x_range, x_range, 1000)
    y = stats.norm.pdf(x, mean, std)
    
    plt.figure(figsize=(10, 6))
    
    # Primary distribution
    plt.plot(x, y, 'b-', linewidth=3, label=f'Normal(μ={mean}, σ={std})', alpha=0.8)
    plt.fill_between(x, y, alpha=0.2, color='blue')
    
    # Comparison distribution
    if comparison and mean2 is not None and std2 is not None:
        y2 = stats.norm.pdf(x, mean2, std2)
        plt.plot(x, y2, 'r-', linewidth=3, label=f'Normal(μ={mean2}, σ={std2})', alpha=0.8)
        plt.fill_between(x, y2, alpha=0.2, color='red')
    
    # Add empirical rule regions (only for primary distribution if not in comparison mode)
    if show_empirical and not comparison:
        # 68% region (±1σ)
        x1 = x[(x >= mean - std) & (x <= mean + std)]
        y1 = stats.norm.pdf(x1, mean, std)
        plt.fill_between(x1, y1, alpha=0.4, color='red', label='68.27% (±1σ)')
        
        # 95% region (±2σ)
        x2 = x[(x >= mean - 2*std) & (x <= mean + 2*std)]
        y2 = stats.norm.pdf(x2, mean, std)
        plt.fill_between(x2, y2, alpha=0.3, color='orange', label='95.45% (±2σ)')
        
        # 99.7% region (±3σ)
        x3 = x[(x >= mean - 3*std) & (x <= mean + 3*std)]
        y3 = stats.norm.pdf(x3, mean, std)
        plt.fill_between(x3, y3, alpha=0.2, color='green', label='99.73% (±3σ)')
    
    # Add percentile lines
    if show_percentiles:
        percentiles = [5, 25, 50, 75, 95]
        colors = ['purple', 'orange', 'red', 'orange', 'purple']
        for p, color in zip(percentiles, colors):
            value = stats.norm.ppf(p/100, mean, std)
            if -x_range <= value <= x_range:
                plt.axvline(value, color=color, linestyle=':', alpha=0.7, 
                           label=f'{p}th percentile')
    
    # Add z-score markers
    if show_zscores and not comparison:
        z_scores = [-3, -2, -1, 0, 1, 2, 3]
        for z in z_scores:
            value = mean + z * std
            if -x_range <= value <= x_range:
                plt.axvline(value, color='gray', linestyle='--', alpha=0.5)
                plt.text(value, max(y) * 0.1, f'z={z}', rotation=90, 
                        ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Mean line
    plt.axvline(mean, color='blue', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'μ₁ = {mean}')
    
    if comparison and mean2 is not None:
        plt.axvline(mean2, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                    label=f'μ₂ = {mean2}')
    
    # Styling
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    
    if comparison:
        plt.title(f'Normal Distribution Comparison', fontsize=14, fontweight='bold')
    else:
        plt.title(f'Normal Distribution (μ={mean}, σ={std})', fontsize=14, fontweight='bold')
    
    # Legend positioning
    if show_empirical or show_percentiles or comparison:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        plt.legend(fontsize=10)
    
    if show_grid:
        plt.grid(True, alpha=0.3)
    
    plt.xlim(-x_range, x_range)
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/generate_uniform_plot')
def generate_uniform_plot():
    a = float(request.args.get('a', 0))
    b = float(request.args.get('b', 1))
    discrete = request.args.get('discrete', 'false').lower() == 'true'
    show_mean = request.args.get('show_mean', 'false').lower() == 'true'
    show_quartiles = request.args.get('show_quartiles', 'false').lower() == 'true'
    show_probabilities = request.args.get('show_probabilities', 'false').lower() == 'true'
    show_cdf = request.args.get('show_cdf', 'false').lower() == 'true'
    show_grid = request.args.get('show_grid', 'true').lower() == 'true'
    comparison = request.args.get('comparison', 'false').lower() == 'true'
    
    # Comparison parameters
    a2 = float(request.args.get('a2', 5)) if comparison else None
    b2 = float(request.args.get('b2', 15)) if comparison else None
    
    # Determine plot layout
    if show_cdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Main distribution plot
    if discrete:
        # Discrete uniform distribution
        values = list(range(int(a), int(b) + 1))
        n = len(values)
        probabilities = [1/n] * n
        
        # Primary distribution
        bars1 = ax1.bar(values, probabilities, alpha=0.7, color='skyblue', 
                        edgecolor='darkblue', width=0.6, label=f'Uniform({int(a)}, {int(b)})')
        
        # Comparison distribution
        if comparison and a2 is not None and b2 is not None:
            values2 = list(range(int(a2), int(b2) + 1))
            n2 = len(values2)
            probabilities2 = [1/n2] * n2
            
            bars2 = ax1.bar(values2, probabilities2, alpha=0.6, color='lightcoral', 
                           edgecolor='darkred', width=0.4, label=f'Uniform({int(a2)}, {int(b2)})')
        
        # Show probability values on bars
        if show_probabilities and not comparison:
            for i, (val, prob) in enumerate(zip(values, probabilities)):
                ax1.text(val, prob + max(probabilities) * 0.02, f'{prob:.3f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set x-axis range
        all_values = values + (values2 if comparison and a2 is not None and b2 is not None else [])
        x_min, x_max = min(all_values) - 1, max(all_values) + 1
        ax1.set_xlim(x_min, x_max)
        ax1.set_xticks(range(x_min, x_max + 1))
        
        ax1.set_xlabel('Value (k)', fontsize=12)
        ax1.set_ylabel('Probability P(X = k)', fontsize=12)
        ax1.set_title('Discrete Uniform Distribution', fontsize=14, fontweight='bold')
        
    else:
        # Continuous uniform distribution
        x_range = max(abs(a) + abs(b), 20)
        if comparison and a2 is not None and b2 is not None:
            x_range = max(x_range, abs(a2) + abs(b2))
        
        x = np.linspace(-x_range, x_range, 1000)
        
        # Primary distribution
        y = np.zeros_like(x)
        mask = (x >= a) & (x <= b)
        y[mask] = 1 / (b - a)
        
        ax1.plot(x, y, 'b-', linewidth=3, label=f'Uniform({a}, {b})', alpha=0.8)
        ax1.fill_between(x, y, alpha=0.3, color='blue')
        
        # Comparison distribution
        if comparison and a2 is not None and b2 is not None:
            y2 = np.zeros_like(x)
            mask2 = (x >= a2) & (x <= b2)
            y2[mask2] = 1 / (b2 - a2)
            
            ax1.plot(x, y2, 'r-', linewidth=3, label=f'Uniform({a2}, {b2})', alpha=0.8)
            ax1.fill_between(x, y2, alpha=0.3, color='red')
        
        # Add boundary lines
        ax1.axvline(a, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f'a = {a}')
        ax1.axvline(b, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f'b = {b}')
        
        if comparison and a2 is not None and b2 is not None:
            ax1.axvline(a2, color='red', linestyle=':', alpha=0.7, linewidth=2, label=f'a₂ = {a2}')
            ax1.axvline(b2, color='red', linestyle=':', alpha=0.7, linewidth=2, label=f'b₂ = {b2}')
        
        ax1.set_xlim(-x_range, x_range)
        ax1.set_xlabel('Value (x)', fontsize=12)
        ax1.set_ylabel('Probability Density f(x)', fontsize=12)
        ax1.set_title('Continuous Uniform Distribution', fontsize=14, fontweight='bold')
    
    # Add mean line
    if show_mean:
        mean = (a + b) / 2
        ax1.axvline(mean, color='green', linestyle='-', alpha=0.8, linewidth=2, 
                   label=f'Mean = {mean:.2f}')
        
        if comparison and a2 is not None and b2 is not None:
            mean2 = (a2 + b2) / 2
            ax1.axvline(mean2, color='orange', linestyle='-', alpha=0.8, linewidth=2, 
                       label=f'Mean₂ = {mean2:.2f}')
    
    # Add quartiles
    if show_quartiles and not discrete:
        q1 = a + 0.25 * (b - a)
        q3 = a + 0.75 * (b - a)
        median = (a + b) / 2
        
        ax1.axvline(q1, color='purple', linestyle='-.', alpha=0.6, label='Q1 (25%)')
        ax1.axvline(median, color='red', linestyle='-.', alpha=0.6, label='Median (50%)')
        ax1.axvline(q3, color='purple', linestyle='-.', alpha=0.6, label='Q3 (75%)')
    
    # Styling
    if show_grid:
        ax1.grid(True, alpha=0.3)
    
    # Legend
    if show_mean or show_quartiles or comparison:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax1.legend(fontsize=10)
    
    # CDF subplot
    if show_cdf:
        if discrete:
            # Discrete CDF
            values = list(range(int(a), int(b) + 1))
            n = len(values)
            cdf_values = [i/n for i in range(1, n+1)]
            
            # Step function for CDF
            x_cdf = []
            y_cdf = []
            
            # Add points before first value
            x_cdf.extend([values[0] - 2, values[0]])
            y_cdf.extend([0, 0])
            
            # Add step points
            for i, (val, cdf_val) in enumerate(zip(values, cdf_values)):
                x_cdf.extend([val, val])
                y_cdf.extend([cdf_values[i-1] if i > 0 else 0, cdf_val])
                if i < len(values) - 1:
                    x_cdf.extend([val, values[i+1]])
                    y_cdf.extend([cdf_val, cdf_val])
            
            # Add points after last value
            x_cdf.extend([values[-1], values[-1] + 2])
            y_cdf.extend([1, 1])
            
            ax2.plot(x_cdf, y_cdf, 'b-', linewidth=2, label='CDF')
            ax2.scatter(values, cdf_values, color='red', s=30, zorder=5)
            
        else:
            # Continuous CDF
            x_cdf = np.linspace(a - 5, b + 5, 1000)
            y_cdf = np.zeros_like(x_cdf)
            
            # Before interval
            y_cdf[x_cdf < a] = 0
            # Within interval
            mask = (x_cdf >= a) & (x_cdf <= b)
            y_cdf[mask] = (x_cdf[mask] - a) / (b - a)
            # After interval
            y_cdf[x_cdf > b] = 1
            
            ax2.plot(x_cdf, y_cdf, 'b-', linewidth=3, label='CDF')
            ax2.axvline(a, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(b, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Value', fontsize=12)
        ax2.set_ylabel('Cumulative Probability F(x)', fontsize=12)
        ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/generate_binomial_plot')
def generate_binomial_plot():
    n = int(request.args.get('n', 10))
    p = float(request.args.get('p', 0.5))
    show_mean = request.args.get('show_mean', 'false').lower() == 'true'
    show_stddev = request.args.get('show_stddev', 'false').lower() == 'true'
    show_normal = request.args.get('show_normal', 'false').lower() == 'true'
    show_cdf = request.args.get('show_cdf', 'false').lower() == 'true'
    show_grid = request.args.get('show_grid', 'true').lower() == 'true'
    comparison = request.args.get('comparison', 'false').lower() == 'true'
    
    # Comparison parameters
    n2 = int(request.args.get('n2', 20)) if comparison else None
    p2 = float(request.args.get('p2', 0.3)) if comparison else None
    
    # Determine plot layout
    if show_cdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate binomial distribution
    x_range = np.arange(0, n + 1)
    y = stats.binom.pmf(x_range, n, p)
    
    # Primary distribution
    bars1 = ax1.bar(x_range, y, alpha=0.7, color='skyblue', edgecolor='darkblue', 
                   width=0.6, label=f'Binomial({n}, {p})')
    
    # Comparison distribution
    if comparison and n2 is not None and p2 is not None:
        x_range2 = np.arange(0, n2 + 1)
        y2 = stats.binom.pmf(x_range2, n2, p2)
        
        bars2 = ax1.bar(x_range2, y2, alpha=0.6, color='lightcoral', edgecolor='darkred', 
                       width=0.4, label=f'Binomial({n2}, {p2})')
    
    # Calculate statistics
    mean = n * p
    variance = n * p * (1 - p)
    std = np.sqrt(variance)
    
    # Show mean line
    if show_mean:
        ax1.axvline(mean, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Mean = {mean:.2f}')
        
        if comparison and n2 is not None and p2 is not None:
            mean2 = n2 * p2
            ax1.axvline(mean2, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
                       label=f'Mean₂ = {mean2:.2f}')
    
    # Show standard deviation regions
    if show_stddev and not comparison:
        # ±1σ region
        x1_lower = max(0, mean - std)
        x1_upper = min(n, mean + std)
        ax1.axvspan(x1_lower, x1_upper, alpha=0.2, color='yellow', label='±1σ')
        
        # ±2σ region
        x2_lower = max(0, mean - 2*std)
        x2_upper = min(n, mean + 2*std)
        ax1.axvspan(x2_lower, x2_upper, alpha=0.1, color='orange', label='±2σ')
    
    # Normal approximation overlay
    if show_normal:
        # Check if normal approximation is appropriate
        np_val = n * p
        nq_val = n * (1 - p)
        
        if np_val >= 5 and nq_val >= 5:
            x_normal = np.linspace(0, n, 1000)
            y_normal = stats.norm.pdf(x_normal, mean, std)
            
            # Scale normal curve to match binomial probabilities
            y_normal_scaled = y_normal * (1.0 / np.max(y_normal)) * np.max(y)
            
            ax1.plot(x_normal, y_normal_scaled, 'r-', linewidth=3, alpha=0.8, 
                    label=f'Normal Approx N({mean:.1f}, {std:.2f})')
    
    # Styling
    max_x = n if not comparison else max(n, n2 if n2 else 0)
    ax1.set_xlim(-1, max_x + 1)
    ax1.set_xlabel('Number of Successes (k)', fontsize=12)
    ax1.set_ylabel('Probability P(X = k)', fontsize=12)
    
    if comparison:
        ax1.set_title('Binomial Distribution Comparison', fontsize=14, fontweight='bold')
    else:
        ax1.set_title(f'Binomial Distribution (n={n}, p={p})', fontsize=14, fontweight='bold')
    
    if show_grid:
        ax1.grid(True, alpha=0.3)
    
    # Legend
    if show_mean or show_stddev or show_normal or comparison:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax1.legend(fontsize=10)
    
    # CDF subplot
    if show_cdf:
        # Calculate CDF
        x_cdf = np.arange(0, n + 1)
        y_cdf = stats.binom.cdf(x_cdf, n, p)
        
        # Step function for CDF
        x_step = []
        y_step = []
        
        # Add initial point
        x_step.extend([-1, 0])
        y_step.extend([0, 0])
        
        # Add step points
        for i, (x_val, cdf_val) in enumerate(zip(x_cdf, y_cdf)):
            x_step.extend([x_val, x_val])
            y_step.extend([y_cdf[i-1] if i > 0 else 0, cdf_val])
            if i < len(x_cdf) - 1:
                x_step.extend([x_val, x_cdf[i+1]])
                y_step.extend([cdf_val, cdf_val])
        
        # Add final point
        x_step.extend([n, n + 1])
        y_step.extend([1, 1])
        
        ax2.plot(x_step, y_step, 'b-', linewidth=2, label='CDF')
        ax2.scatter(x_cdf, y_cdf, color='red', s=30, zorder=5)
        
        # Comparison CDF
        if comparison and n2 is not None and p2 is not None:
            x_cdf2 = np.arange(0, n2 + 1)
            y_cdf2 = stats.binom.cdf(x_cdf2, n2, p2)
            
            # Step function for comparison CDF
            x_step2 = []
            y_step2 = []
            
            x_step2.extend([-1, 0])
            y_step2.extend([0, 0])
            
            for i, (x_val, cdf_val) in enumerate(zip(x_cdf2, y_cdf2)):
                x_step2.extend([x_val, x_val])
                y_step2.extend([y_cdf2[i-1] if i > 0 else 0, cdf_val])
                if i < len(x_cdf2) - 1:
                    x_step2.extend([x_val, x_cdf2[i+1]])
                    y_step2.extend([cdf_val, cdf_val])
            
            x_step2.extend([n2, max(n, n2) + 1])
            y_step2.extend([1, 1])
            
            ax2.plot(x_step2, y_step2, 'r-', linewidth=2, alpha=0.7, label=f'CDF₂')
            ax2.scatter(x_cdf2, y_cdf2, color='orange', s=30, zorder=5)
        
        ax2.set_xlabel('Number of Successes (k)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability F(k)', fontsize=12)
        ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
        ax2.set_xlim(-1, max_x + 1)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/coin_toss_simulation')
def coin_toss_simulation():
    n_tosses = int(request.args.get('n_tosses', 100))
    
    # Simulate coin tosses
    tosses = np.random.choice(['H', 'T'], size=n_tosses, p=[0.5, 0.5])
    
    # Count heads in groups
    group_size = 10
    n_groups = n_tosses // group_size
    heads_counts = []
    
    for i in range(n_groups):
        group = tosses[i * group_size:(i + 1) * group_size]
        heads_count = np.sum(group == 'H')
        heads_counts.append(heads_count)
    
    # Calculate cumulative proportion of heads
    cumulative_heads = np.cumsum(tosses == 'H')
    cumulative_tosses = np.arange(1, n_tosses + 1)
    cumulative_proportion = cumulative_heads / cumulative_tosses
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Individual tosses
    plt.subplot(2, 2, 1)
    colors = ['red' if toss == 'H' else 'blue' for toss in tosses[:50]]  # Show first 50 tosses
    plt.bar(range(min(50, len(tosses))), [1] * min(50, len(tosses)), color=colors, alpha=0.7)
    plt.title(f'First {min(50, len(tosses))} Coin Tosses')
    plt.xlabel('Toss Number')
    plt.ylabel('Outcome')
    plt.yticks([0.5, 1], ['', 'H/T'])
    
    # Subplot 2: Heads count in groups
    plt.subplot(2, 2, 2)
    plt.bar(range(len(heads_counts)), heads_counts, alpha=0.7, color='green')
    plt.axhline(y=5, color='red', linestyle='--', label='Expected (5)')
    plt.title(f'Heads Count per {group_size} Tosses')
    plt.xlabel('Group Number')
    plt.ylabel('Number of Heads')
    plt.legend()
    
    # Subplot 3: Cumulative proportion
    plt.subplot(2, 2, 3)
    plt.plot(cumulative_tosses, cumulative_proportion, 'b-', alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Expected (0.5)')
    plt.title('Cumulative Proportion of Heads')
    plt.xlabel('Number of Tosses')
    plt.ylabel('Proportion of Heads')
    plt.legend()
    plt.ylim(0, 1)
    
    # Subplot 4: Histogram of group results
    plt.subplot(2, 2, 4)
    plt.hist(heads_counts, bins=range(12), alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Heads Count')
    plt.xlabel('Number of Heads')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({
        'plot': plot_url,
        'total_heads': int(np.sum(tosses == 'H')),
        'proportion': float(np.sum(tosses == 'H') / n_tosses),
        'tosses_sequence': ''.join(tosses[:100])  # Return first 100 tosses
    })

@app.route('/sample_mean_simulation')
def sample_mean_simulation():
    sample_size = int(request.args.get('sample_size', 30))
    num_samples = int(request.args.get('num_samples', 100))
    distribution = request.args.get('distribution', 'normal')
    
    sample_means = []
    
    for _ in range(num_samples):
        if distribution == 'normal':
            sample = np.random.normal(50, 10, sample_size)
        elif distribution == 'uniform':
            sample = np.random.uniform(0, 100, sample_size)
        elif distribution == 'exponential':
            sample = np.random.exponential(2, sample_size)
        else:  # skewed
            sample = np.random.gamma(2, 2, sample_size)
        
        sample_means.append(np.mean(sample))
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Original distribution
    plt.subplot(2, 2, 1)
    if distribution == 'normal':
        x = np.linspace(20, 80, 1000)
        y = stats.norm.pdf(x, 50, 10)
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Original Distribution: Normal(50, 10)')
    elif distribution == 'uniform':
        x = np.linspace(0, 100, 1000)
        y = np.ones_like(x) / 100
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Original Distribution: Uniform(0, 100)')
    elif distribution == 'exponential':
        x = np.linspace(0, 20, 1000)
        y = stats.expon.pdf(x, scale=2)
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Original Distribution: Exponential(λ=0.5)')
    else:
        x = np.linspace(0, 20, 1000)
        y = stats.gamma.pdf(x, 2, scale=2)
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Original Distribution: Gamma(2, 2)')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # Subplot 2: Sample from original distribution
    plt.subplot(2, 2, 2)
    if distribution == 'normal':
        sample_data = np.random.normal(50, 10, 1000)
    elif distribution == 'uniform':
        sample_data = np.random.uniform(0, 100, 1000)
    elif distribution == 'exponential':
        sample_data = np.random.exponential(2, 1000)
    else:
        sample_data = np.random.gamma(2, 2, 1000)
    
    plt.hist(sample_data, bins=30, alpha=0.7, density=True, color='lightblue')
    plt.title(f'Sample from Original Distribution (n=1000)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # Subplot 3: Distribution of sample means
    plt.subplot(2, 2, 3)
    plt.hist(sample_means, bins=20, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
    
    # Overlay normal curve based on CLT
    if distribution == 'normal':
        theoretical_mean = 50
        theoretical_std = 10 / np.sqrt(sample_size)
    elif distribution == 'uniform':
        theoretical_mean = 50
        theoretical_std = (100 / np.sqrt(12)) / np.sqrt(sample_size)
    elif distribution == 'exponential':
        theoretical_mean = 2
        theoretical_std = 2 / np.sqrt(sample_size)
    else:  # gamma
        theoretical_mean = 4
        theoretical_std = (2 * np.sqrt(2)) / np.sqrt(sample_size)
    
    x_norm = np.linspace(min(sample_means), max(sample_means), 100)
    y_norm = stats.norm.pdf(x_norm, theoretical_mean, theoretical_std)
    plt.plot(x_norm, y_norm, 'r-', linewidth=2, label='Theoretical Normal')
    
    plt.title(f'Distribution of Sample Means (n={sample_size}, samples={num_samples})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    
    # Subplot 4: Q-Q plot
    plt.subplot(2, 2, 4)
    stats.probplot(sample_means, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Sample Means vs Normal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({
        'plot': plot_url,
        'sample_mean_of_means': float(np.mean(sample_means)),
        'sample_std_of_means': float(np.std(sample_means)),
        'theoretical_mean': float(theoretical_mean),
        'theoretical_std': float(theoretical_std)
    })

@app.route('/confidence_interval_simulation')
def confidence_interval_simulation():
    population_mean = float(request.args.get('population_mean', 100))
    population_std = float(request.args.get('population_std', 15))
    sample_size = int(request.args.get('sample_size', 30))
    confidence_level = float(request.args.get('confidence_level', 95))
    num_intervals = int(request.args.get('num_intervals', 100))
    
    # Calculate critical value
    alpha = (100 - confidence_level) / 100
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    intervals = []
    contains_true_mean = []
    
    for i in range(num_intervals):
        # Generate a sample from the population
        sample = np.random.normal(population_mean, population_std, sample_size)
        sample_mean = np.mean(sample)
        
        # Calculate margin of error
        margin_error = z_critical * (population_std / np.sqrt(sample_size))
        
        # Calculate confidence interval
        lower_bound = sample_mean - margin_error
        upper_bound = sample_mean + margin_error
        
        intervals.append((lower_bound, upper_bound))
        contains_true_mean.append(lower_bound <= population_mean <= upper_bound)
    
    # Calculate coverage probability
    coverage_probability = np.mean(contains_true_mean)
    
    plt.figure(figsize=(12, 8))
    
    # Plot confidence intervals
    for i, (lower, upper) in enumerate(intervals[:50]):  # Show first 50 intervals
        color = 'green' if contains_true_mean[i] else 'red'
        alpha = 0.7 if contains_true_mean[i] else 0.9
        plt.plot([lower, upper], [i, i], color=color, alpha=alpha, linewidth=2)
        plt.plot([(lower + upper)/2], [i], 'o', color=color, markersize=3)
    
    # Add vertical line for true population mean
    plt.axvline(population_mean, color='blue', linestyle='--', linewidth=2, 
                label=f'True Population Mean = {population_mean}')
    
    plt.xlabel('Value')
    plt.ylabel('Confidence Interval Number')
    plt.title(f'{confidence_level}% Confidence Intervals\n'
              f'Coverage: {coverage_probability:.1%} (Expected: {confidence_level}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Sample Size: {sample_size}\n' \
              f'Population μ: {population_mean}\n' \
              f'Population σ: {population_std}\n' \
              f'Intervals shown: {min(50, num_intervals)}/{num_intervals}\n' \
              f'Coverage rate: {coverage_probability:.1%}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({
        'plot': plot_url,
        'coverage_probability': float(coverage_probability),
        'expected_coverage': float(confidence_level / 100),
        'num_containing_mean': int(sum(contains_true_mean)),
        'total_intervals': num_intervals,
        'margin_error': float(margin_error)
    })

@app.route('/generate_sampling_plot')
def generate_sampling_plot():
    # Get parameters from request
    distribution = request.args.get('distribution', 'normal')
    mean = float(request.args.get('mean', 50))
    std = float(request.args.get('std', 10))
    sample_size = int(request.args.get('sample_size', 30))
    num_samples = int(request.args.get('num_samples', 1000))
    
    # Display options
    show_population = request.args.get('show_population', 'true').lower() == 'true'
    show_sampling = request.args.get('show_sampling', 'true').lower() == 'true'
    show_normal = request.args.get('show_normal', 'false').lower() == 'true'
    show_grid = request.args.get('show_grid', 'true').lower() == 'true'
    
    # Comparison and CLT modes
    comparison = request.args.get('comparison', 'false').lower() == 'true'
    clt_mode = request.args.get('clt_mode', 'false').lower() == 'true'
    
    # Comparison parameters
    n2 = int(request.args.get('n2', 10)) if comparison else None
    n3 = int(request.args.get('n3', 50)) if comparison else None
    
    # CLT parameters
    clt_sizes = []
    clt_distribution = 'uniform'
    if clt_mode:
        clt_sizes_str = request.args.get('clt_sizes', '5,10,30')
        clt_sizes = [int(x) for x in clt_sizes_str.split(',') if x.strip()]
        clt_distribution = request.args.get('clt_distribution', 'uniform')
    
    # Determine subplot layout
    if clt_mode:
        # CLT demo: show population + multiple sampling distributions
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, len(clt_sizes), height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        pop_ax = fig.add_subplot(gs[0, :])
        sample_axes = [fig.add_subplot(gs[1, i]) for i in range(len(clt_sizes))]
    elif comparison:
        # Comparison mode: population + 3 sampling distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sample Size Comparison - Sampling Distributions', fontsize=16, fontweight='bold')
        pop_ax = axes[0, 0]
        sample_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    else:
        # Standard mode: population + sampling distribution
        if show_population and show_sampling:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            pop_ax, sample_ax = ax1, ax2
        elif show_population:
            fig, pop_ax = plt.subplots(1, 1, figsize=(10, 6))
            sample_ax = None
        else:
            fig, sample_ax = plt.subplots(1, 1, figsize=(10, 6))
            pop_ax = None
    
    # Generate population data
    def generate_population_data(dist_type, mu, sigma, size=10000):
        np.random.seed(42)  # For reproducibility
        
        if dist_type == 'normal':
            return np.random.normal(mu, sigma, size)
        elif dist_type == 'uniform':
            # For uniform, adjust bounds to match desired mean and std
            a = mu - sigma * np.sqrt(3)
            b = mu + sigma * np.sqrt(3)
            return np.random.uniform(a, b, size)
        elif dist_type == 'exponential':
            # Exponential with desired mean
            scale = mu
            data = np.random.exponential(scale, size)
            # Adjust to match desired std (approximately)
            return data * (sigma / np.std(data)) + (mu - np.mean(data * (sigma / np.std(data))))
        elif dist_type == 'gamma':
            # Gamma distribution with shape=2 for moderate skew
            shape = 2
            scale = sigma**2 / mu  # Adjust scale to get desired mean/std relationship
            data = np.random.gamma(shape, scale, size)
            # Normalize to desired mean and std
            return (data - np.mean(data)) * (sigma / np.std(data)) + mu
        elif dist_type == 'bimodal':
            # Mixture of two normals
            data1 = np.random.normal(mu - sigma, sigma/2, size//2)
            data2 = np.random.normal(mu + sigma, sigma/2, size - size//2)
            data = np.concatenate([data1, data2])
            np.random.shuffle(data)
            return data
        else:
            return np.random.normal(mu, sigma, size)
    
    # Generate sampling distribution
    def generate_sampling_distribution(pop_data, n, num_samples):
        np.random.seed(123)  # Different seed for sampling
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(pop_data, size=n, replace=True)
            sample_means.append(np.mean(sample))
        return np.array(sample_means)
    
    # Plot population distribution
    if show_population and pop_ax is not None:
        if clt_mode:
            pop_data = generate_population_data(clt_distribution, mean, std)
            dist_name = clt_distribution.capitalize()
        else:
            pop_data = generate_population_data(distribution, mean, std)
            dist_name = distribution.capitalize()
        
        # Create histogram
        pop_ax.hist(pop_data, bins=50, density=True, alpha=0.7, color='lightblue', 
                   edgecolor='darkblue', label=f'{dist_name} Population')
        
        # Add theoretical curve if normal
        if (clt_mode and clt_distribution == 'normal') or (not clt_mode and distribution == 'normal'):
            x_range = np.linspace(np.min(pop_data), np.max(pop_data), 1000)
            y_theoretical = stats.norm.pdf(x_range, mean, std)
            pop_ax.plot(x_range, y_theoretical, 'r-', linewidth=2, 
                       label=f'Theoretical Normal({mean}, {std})')
        
        pop_ax.axvline(np.mean(pop_data), color='red', linestyle='--', linewidth=2, 
                      label=f'Population μ = {np.mean(pop_data):.1f}')
        
        pop_ax.set_title(f'{dist_name} Population Distribution\n(μ={mean}, σ={std})', 
                        fontsize=12, fontweight='bold')
        pop_ax.set_xlabel('Value', fontsize=10)
        pop_ax.set_ylabel('Density', fontsize=10)
        pop_ax.legend(fontsize=9)
        if show_grid:
            pop_ax.grid(True, alpha=0.3)
    
    # Generate and plot sampling distributions
    if clt_mode:
        # CLT demonstration
        if clt_distribution == 'uniform':
            pop_data = generate_population_data('uniform', mean, std)
        elif clt_distribution == 'exponential':
            pop_data = generate_population_data('exponential', mean, std)
        else:  # bimodal
            pop_data = generate_population_data('bimodal', mean, std)
        
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for i, n in enumerate(clt_sizes):
            if i < len(sample_axes):
                ax = sample_axes[i]
                
                # Generate sampling distribution
                sample_means = generate_sampling_distribution(pop_data, n, num_samples)
                
                # Calculate statistics
                sample_mean = np.mean(sample_means)
                sample_std = np.std(sample_means)
                theoretical_se = std / np.sqrt(n)
                
                # Plot histogram
                ax.hist(sample_means, bins=30, density=True, alpha=0.7, 
                       color=colors[i % len(colors)], edgecolor='black',
                       label=f'Sample Means (n={n})')
                
                # Normal overlay
                x_range = np.linspace(np.min(sample_means), np.max(sample_means), 1000)
                y_normal = stats.norm.pdf(x_range, mean, theoretical_se)
                ax.plot(x_range, y_normal, 'k-', linewidth=2, 
                       label=f'Normal({mean}, {theoretical_se:.2f})')
                
                # Mean line
                ax.axvline(sample_mean, color='red', linestyle='--', linewidth=2,
                          label=f'Sample Mean = {sample_mean:.2f}')
                
                ax.set_title(f'Sampling Distribution (n={n})\nSE = {sample_std:.3f} (Theory: {theoretical_se:.3f})', 
                            fontsize=10, fontweight='bold')
                ax.set_xlabel('Sample Mean', fontsize=9)
                ax.set_ylabel('Density', fontsize=9)
                ax.legend(fontsize=8)
                if show_grid:
                    ax.grid(True, alpha=0.3)
                
                # Adjust y-axis to show normal shape clearly
                ax.set_ylim(0, np.max(y_normal) * 1.2)
    
    elif comparison:
        # Comparison mode
        pop_data = generate_population_data(distribution, mean, std)
        sample_sizes = [sample_size, n2, n3]
        colors = ['blue', 'red', 'green']
        
        for i, n in enumerate(sample_sizes):
            if i < len(sample_axes):
                ax = sample_axes[i]
                
                # Generate sampling distribution
                sample_means = generate_sampling_distribution(pop_data, n, num_samples)
                
                # Calculate statistics
                sample_mean = np.mean(sample_means)
                sample_std = np.std(sample_means)
                theoretical_se = std / np.sqrt(n)
                
                # Plot histogram
                ax.hist(sample_means, bins=30, density=True, alpha=0.7, 
                       color=colors[i], edgecolor='black',
                       label=f'Sample Means (n={n})')
                
                # Normal overlay if requested
                if show_normal:
                    x_range = np.linspace(np.min(sample_means), np.max(sample_means), 1000)
                    y_normal = stats.norm.pdf(x_range, mean, theoretical_se)
                    ax.plot(x_range, y_normal, 'k-', linewidth=2, 
                           label=f'Normal Overlay')
                
                # Mean line
                ax.axvline(sample_mean, color='red', linestyle='--', linewidth=2,
                          label=f'Mean = {sample_mean:.2f}')
                
                # Standard error regions
                ax.axvspan(sample_mean - sample_std, sample_mean + sample_std, 
                          alpha=0.2, color=colors[i], label='±1 SE')
                
                ax.set_title(f'Sample Size n = {n}\nSE = {sample_std:.3f} (Theory: {theoretical_se:.3f})', 
                            fontsize=11, fontweight='bold')
                ax.set_xlabel('Sample Mean', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.legend(fontsize=8)
                if show_grid:
                    ax.grid(True, alpha=0.3)
        
        # Population plot styling
        pop_ax.set_title(f'{distribution.capitalize()} Population\n(Source for all samples)', 
                        fontsize=11, fontweight='bold')
    
    else:
        # Standard mode
        if show_sampling and ('sample_ax' in locals() or not show_population):
            if not show_population:
                sample_ax = pop_ax if pop_ax else plt.gca()
            
            pop_data = generate_population_data(distribution, mean, std)
            sample_means = generate_sampling_distribution(pop_data, sample_size, num_samples)
            
            # Calculate statistics
            sample_mean = np.mean(sample_means)
            sample_std = np.std(sample_means)
            theoretical_se = std / np.sqrt(sample_size)
            
            # Plot sampling distribution
            sample_ax.hist(sample_means, bins=40, density=True, alpha=0.7, color='lightgreen', 
                          edgecolor='darkgreen', label=f'Sample Means (n={sample_size})')
            
            # Normal overlay
            if show_normal:
                x_range = np.linspace(np.min(sample_means), np.max(sample_means), 1000)
                y_normal = stats.norm.pdf(x_range, mean, theoretical_se)
                sample_ax.plot(x_range, y_normal, 'r-', linewidth=3, 
                              label=f'Normal({mean}, {theoretical_se:.2f})')
            
            # Mean and standard error
            sample_ax.axvline(sample_mean, color='red', linestyle='--', linewidth=2, 
                             label=f'Sample Mean = {sample_mean:.2f}')
            
            # Standard error regions
            sample_ax.axvspan(sample_mean - sample_std, sample_mean + sample_std, 
                             alpha=0.2, color='yellow', label='±1 SE')
            sample_ax.axvspan(sample_mean - 2*sample_std, sample_mean + 2*sample_std, 
                             alpha=0.1, color='orange', label='±2 SE')
            
            sample_ax.set_title(f'Sampling Distribution of Sample Means\n'
                               f'n={sample_size}, {num_samples} samples, SE={sample_std:.3f}', 
                               fontsize=12, fontweight='bold')
            sample_ax.set_xlabel('Sample Mean', fontsize=10)
            sample_ax.set_ylabel('Density', fontsize=10)
            sample_ax.legend(fontsize=9)
            if show_grid:
                sample_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

if __name__ == '__main__':
    app.run(debug=True, port=5000)