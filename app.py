from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)

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

@app.route('/central_limit_theorem')
def central_limit_theorem():
    return render_template('central_limit_theorem.html')

@app.route('/estimations')
def estimations():
    return render_template('estimations.html')

@app.route('/generate_normal_plot')
def generate_normal_plot():
    mean = float(request.args.get('mean', 0))
    std = float(request.args.get('std', 1))
    
    # Fixed x-axis range from -10 to 10 for consistent viewing
    x = np.linspace(-10, 10, 1000)
    y = stats.norm.pdf(x, mean, std)
    
    plt.figure(figsize=(8, 4.5))  # Reduced figure size from (10, 6) to (8, 4.5)
    plt.plot(x, y, 'b-', linewidth=2, label=f'Normal(μ={mean}, σ={std})')
    plt.fill_between(x, y, alpha=0.3)
    
    # Add empirical rule regions for standard normal
    if mean == 0 and std == 1:
        # 68% region (±1σ)
        x1 = x[(x >= -1) & (x <= 1)]
        y1 = stats.norm.pdf(x1, mean, std)
        plt.fill_between(x1, y1, alpha=0.5, color='red', label='68.27% (±1σ)')
        
        # 95% region (±2σ)
        x2 = x[(x >= -2) & (x <= 2)]
        y2 = stats.norm.pdf(x2, mean, std)
        plt.fill_between(x2, y2, alpha=0.3, color='blue', label='95.45% (±2σ)')
        
        # 99.7% region (±3σ)
        x3 = x[(x >= -3) & (x <= 3)]
        y3 = stats.norm.pdf(x3, mean, std)
        plt.fill_between(x3, y3, alpha=0.2, color='purple', label='99.73% (±3σ)')
    
    plt.axvline(mean, color='red', linestyle='--', alpha=0.7, label=f'Mean = {mean}')
    plt.xlabel('Value', fontsize=11)  # Reduced font size
    plt.ylabel('Probability Density', fontsize=11)  # Reduced font size
    plt.title(f'Normal Distribution (μ={mean}, σ={std})', fontsize=12)  # Reduced font size
    plt.legend(fontsize=9)  # Reduced legend font size
    plt.grid(True, alpha=0.3)
    
    # Set fixed x-axis limits
    plt.xlim(-10, 10)
    
    # Adjust layout to be more compact
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=120)  # Reduced DPI from 150 to 120
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/generate_uniform_plot')
def generate_uniform_plot():
    a = float(request.args.get('a', 0))
    b = float(request.args.get('b', 1))
    discrete = request.args.get('discrete', 'false').lower() == 'true'
    
    plt.figure(figsize=(8, 4.5))  # Reduced figure size
    
    if discrete:
        # Discrete uniform distribution with fixed x-axis range
        values = list(range(int(a), int(b) + 1))
        probabilities = [1/len(values)] * len(values)
        
        # Create bars only for the active values
        plt.bar(values, probabilities, alpha=0.7, color='skyblue', edgecolor='black', width=0.8)
        
        # Set fixed x-axis limits and ticks
        plt.xlim(-1, 16)
        plt.xticks(range(0, 16))
        plt.xlabel('Value', fontsize=11)  # Reduced font size
        plt.ylabel('Probability', fontsize=11)  # Reduced font size
        plt.title(f'Discrete Uniform Distribution (a={int(a)}, b={int(b)})', fontsize=12)  # Reduced font size
        
        # Add probability value labels on top of bars
        for i, (val, prob) in enumerate(zip(values, probabilities)):
            plt.text(val, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontsize=8)  # Reduced font size
        
        # Set y-axis limit to accommodate text labels
        max_prob = max(probabilities) if probabilities else 0.5
        plt.ylim(0, max_prob + 0.05)
    else:
        # Continuous uniform distribution
        x = np.linspace(-1, 16, 1000)
        y = np.zeros_like(x)
        
        # Set probability density for the interval [a, b]
        mask = (x >= a) & (x <= b)
        y[mask] = 1 / (b - a)
        
        plt.plot(x, y, 'b-', linewidth=2, label=f'Uniform({a}, {b})')
        plt.fill_between(x, y, alpha=0.3)
        
        # Add vertical lines at boundaries
        plt.axvline(a, color='red', linestyle='--', alpha=0.7, label=f'a = {a}')
        plt.axvline(b, color='green', linestyle='--', alpha=0.7, label=f'b = {b}')
        
        plt.xlim(-1, 16)
        plt.xlabel('Value', fontsize=11)
        plt.ylabel('Probability Density', fontsize=11)
        plt.title(f'Continuous Uniform Distribution (a={a}, b={b})', fontsize=12)
        plt.legend(fontsize=9)
        
        plt.ylim(0, max(y) + 0.1 if max(y) > 0 else 0.5)
    
    plt.grid(True, alpha=0.3)
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
    
    # Fixed x-axis range from 0 to 50 to accommodate all possible slider values
    x_range = np.arange(0, 51)
    x_active = np.arange(0, n + 1)
    y_active = stats.binom.pmf(x_active, n, p)
    
    plt.figure(figsize=(8, 4.5))
    
    # Create bars only for the active range (0 to n)
    plt.bar(x_active, y_active, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # Set fixed x-axis limits and labels
    plt.xlim(-1, 51)
    plt.xticks(range(0, 51, 5))  # Show ticks every 5 units for readability
    plt.xlabel('Number of Successes (k)', fontsize=11)
    plt.ylabel('Probability P(X = k)', fontsize=11)
    plt.title(f'Binomial Distribution (n={n}, p={p})', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add mean and variance info
    mean = n * p
    variance = n * p * (1 - p)
    plt.axvline(mean, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean = {mean:.2f}\nVariance = {variance:.2f}')
    plt.legend(fontsize=9)
    
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)