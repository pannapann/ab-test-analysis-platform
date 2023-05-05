# A/B Test Tracking Streamlit App

This repository contains a Streamlit app for tracking and analyzing A/B tests. The app allows users to upload their A/B test data in CSV format and provides an interactive interface for analyzing the data, calculating effect sizes, required sample sizes, and performing various statistical tests.

## Demo
https://youtu.be/L021DyIxJfw

## Features

The app supports the following features:

- Upload A/B test data in CSV format
- Select columns for date, group, and target
- Calculate effect size for t-test, proportional z-test, chi-square test, or Mann-Whitney U test
- Calculate required sample size for given test type, effect size, power, and significance level
- Estimate number of days needed to collect required sample size for given test type, effect size, power, and significance level
- Perform statistical tests and display results
- Perform Bayesian analysis
- Plot target by date
- Plot target by segment
- Plot statistical significance
- Check A/B group bias
- Plot cumulative metrics
- Plot metric distributions
- Analyze seasonality
- Plot cumulative target


### Installation

1. Clone this repository:
```bash
git clone https://github.com/pannapann/ab-test-analysis-platform
```

2. Change to the repository directory:
```bash
cd ab-test-tracking
```
3. Create a virtual environment:
```bash
python -m venv venv
```

4. Activate the virtual environment:
On Windows:
```bash
venv\Scripts\activate
```

On macOS and Linux:
```bash
source venv/bin/activate
```

5. Install required packages:
```bash
pip install -r requirements.txt
```



### Usage

1. Start the Streamlit web application:
```bash
streamlit run app.py
```

2. Open the web application in your browser using the URL provided in the terminal.

3. Upload your A/B test data in CSV format and follow the prompts to analyze your data and visualize the results.






