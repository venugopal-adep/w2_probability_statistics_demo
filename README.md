# Probability Distributions Learning App

**Developed by Venugopal Adep**

An interactive web-based learning platform for understanding probability distributions, random variables, and statistical inference concepts. This educational application provides hands-on visualization and exploration tools for students and educators in statistics and data science.

## 🎯 What is this Application?

This Flask-based web application offers an intuitive way to learn and understand fundamental concepts in probability and statistics through interactive visualizations and demonstrations. The app covers:

### Core Topics Covered:
- **Random Variables** - Discrete and continuous random variables
- **Probability Distributions** - Normal, Uniform, and Binomial distributions
- **Sampling Distributions** - Central Limit Theorem demonstrations
- **Statistical Estimation** - Confidence intervals and parameter estimation
- **Hypothesis Testing** - Type I/II errors, p-values, and power analysis

### Key Features:
- 🎨 **Interactive Visualizations** - Real-time plot generation with customizable parameters
- 📊 **Multiple Distribution Types** - Comprehensive coverage of common distributions
- 🎯 **Educational Quizzes** - Built-in assessment tools with immediate feedback
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile devices
- 🎪 **Simulation Tools** - Monte Carlo simulations and sampling demonstrations
- 📈 **Statistical Analysis** - Real-time statistical calculations and interpretations

## 🚀 Getting Started

### Prerequisites

Before running this application, ensure you have the following installed on your machine:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package installer) - Usually comes with Python
- **Git** - [Download Git](https://git-scm.com/downloads)

### Installation & Setup

#### Method 1: Clone from GitHub (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/venugopal-adep/w2_probability_statistics_demo.git
   cd w2_probability_statistics_demo
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

#### Method 2: Download ZIP

1. Download the ZIP file from the GitHub repository
2. Extract the files to your desired location
3. Follow steps 2-5 from Method 1

### 📦 Dependencies

The application requires the following Python packages (automatically installed via requirements.txt):

```
Flask==2.3.3
Flask-CORS==4.0.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
```

## 🎮 How to Use the Application

### Navigation

The application features an intuitive navigation system with the following sections:

1. **Home** - Overview and introduction to probability distributions
2. **Distributions Menu**:
   - Random Variables
   - Normal Distribution
   - Uniform Distribution  
   - Binomial Distribution
3. **Inference Menu**:
   - Sampling Distributions & CLT
   - Estimations
   - Hypothesis Testing

### Interactive Features

#### 🎯 Distribution Explorers
- Adjust parameters using sliders and input fields
- See real-time updates to probability density/mass functions
- Compare multiple distributions side-by-side
- Visualize empirical rules and percentiles

#### 📊 Simulation Tools
- Coin toss simulations
- Sample mean demonstrations
- Central Limit Theorem visualizations
- Confidence interval simulations

#### 🧠 Educational Quizzes
- Interactive multiple-choice questions
- Immediate feedback with explanations
- True/False format for quick assessment
- Progress tracking

## 🔧 Project Structure

```
W2_Statistics_Demo1/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── static/
│   └── plots/                      # Generated plot storage
├── templates/
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Home page
│   ├── random_variable.html        # Random variables page
│   ├── normal.html                 # Normal distribution page
│   ├── uniform.html                # Uniform distribution page
│   ├── binomial.html               # Binomial distribution page
│   ├── sampling_distributions.html # Sampling & CLT page
│   ├── estimations.html            # Statistical estimation page
│   ├── hypothesis_testing.html     # Hypothesis testing page
│   └── central_limit_theorem.html  # CLT demonstration page
└── __pycache__/                    # Python cache files
```

## 🎨 Features Overview

### Normal Distribution Explorer
- Interactive parameter adjustment (mean, standard deviation)
- Empirical rule visualization (68-95-99.7 rule)
- Z-score calculations and percentiles
- Comparison mode for multiple distributions

### Uniform Distribution Visualizer
- Both discrete and continuous uniform distributions
- Probability mass/density function displays
- Cumulative distribution function (CDF) plots
- Real-world examples and applications

### Binomial Distribution Simulator
- Success/failure trial simulations
- Normal approximation comparisons
- Interactive parameter exploration (n, p)
- Real-time probability calculations

### Central Limit Theorem Demonstration
- Multiple sample size comparisons
- Various population distribution shapes
- Sampling distribution visualizations
- Statistical convergence illustrations

### Hypothesis Testing Tools
- Type I and Type II error visualizations
- Power analysis demonstrations
- Critical region illustrations
- P-value interpretations

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **Port 5000 already in use**
   ```bash
   # Kill the process using port 5000
   sudo lsof -t -i:5000 | xargs sudo kill
   
   # Or run on a different port
   python app.py --port 5001
   ```

2. **Module not found errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. **Plot generation issues**
   - Ensure matplotlib backend is properly configured
   - Check write permissions in the static/plots directory

4. **Browser compatibility**
   - Use modern browsers (Chrome, Firefox, Safari, Edge)
   - Enable JavaScript for full functionality

## 🛠️ Development

### Running in Development Mode

```bash
# Set Flask environment variables
export FLASK_ENV=development  # macOS/Linux
set FLASK_ENV=development     # Windows

# Run with auto-reload
python app.py
```

### Adding New Features

The application is built with modularity in mind. To add new distributions or features:

1. Create new route handlers in `app.py`
2. Add corresponding HTML templates in `templates/`
3. Update navigation in `base.html`
4. Add any new dependencies to `requirements.txt`

## 📱 Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

**Venugopal Adep**
- GitHub: [@venugopal-adep](https://github.com/venugopal-adep)

## 🙏 Acknowledgments

- Built with Flask, NumPy, Matplotlib, and SciPy
- Bootstrap for responsive UI design
- Font Awesome for icons
- Educational content inspired by modern statistics curricula

---

### 🌟 Star this repository if you find it helpful!

**Happy Learning! 📊📈**