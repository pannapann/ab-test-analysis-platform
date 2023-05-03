import streamlit as st
import pandas as pd
from main import ABtestTrackingClass

# Import any additional dependencies required for ABtestTrackingClass, e.g., Plotly, numpy, and statsmodels

# Create a Streamlit app
st.set_page_config(page_title="A/B Test Tracking", layout="wide")

st.title("A/B Test Tracking")

# Upload data
uploaded_file = st.sidebar.file_uploader("Upload your A/B test data (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Get column names for user input
    date_column = st.sidebar.selectbox("Select the date column", data.columns)
    group_column = st.sidebar.selectbox("Select the group column", data.columns)
    target_column = st.sidebar.selectbox("Select the target column", data.columns)

    # Create an ABTestTracking instance
    ab_test = ABtestTrackingClass(data, target_column, date_column, group_column)

    # Display dataframe
    st.write("Data:")
    st.write(ab_test.dataframe.head())

    # Calculate effect size
    test_type = st.selectbox("Select a test type", options=['t_test', 'proportional_z', 'chi_square', 'mann_whitney_u'])
    st.write("Effect Size:", ab_test.calculate_effect_size(test_type=test_type))

    # Calculate required sample size
    effect_size = st.number_input("Enter the effect size", value=0.1, step=0.01)
    power = st.number_input("Enter the power (between 0 and 1)", value=0.8, step=0.01)
    significance_level = st.number_input("Enter the significance level (between 0 and 1)", value=0.05, step=0.01)
    st.write("Required Sample Size:", ab_test.calculate_required_sample_size(test_type=test_type, effect_size=effect_size, power=power, alpha=significance_level))

    # Estimate number of days needed to collect required sample size
    st.write("Estimated Number of Days Needed:", ab_test.estimate_days_to_collect_data(test_type=test_type, effect_size=effect_size, power=power, alpha=significance_level))

    # Perform statistical tests
    if test_type == "chi_square":
        st.write("Chi-square Test Results:", ab_test.chi_square_test())
    elif test_type == "proportional_z":
        st.write("Proportional Z-test Results:", ab_test.proportional_z_test())
    elif test_type == "t_test":
        st.write("T-test Results:", ab_test.t_test())
    elif test_type == "mann_whitney_u":
        st.write("Mann whitney u Test Results:", ab_test.mann_whitney_u_test())
    else:
        st.write("Frequentist Mean-based Test Results:", ab_test.frequentist_mean_based(test_type=test_type))

    # Perform Bayesian analysis
    if ab_test.dataframe[ab_test.target_column].unique().size > 2:
        ab_test.bayesian_mean_based()
    else:
        ab_test.bayesian_proportion_based()
        

    # Plot target by date
    st.subheader("Plot Target by Date")
    aggregation_level = st.selectbox("Select aggregation level", options=["daily", "weekly", "monthly"])
    plot_type = st.selectbox("Select plot type", options=["proportion_based", "mean_based"])
    ab_test.plot_target_by_date(aggregation=aggregation_level, plot_type=plot_type)

    # Plot target by segment
    st.subheader("Plot Target by Segment")
    segment_column = st.selectbox("Select a segment column", data.columns)
    plot_type = st.selectbox("Select plot type for segment", options=["proportion_based", "mean_based"])
    ab_test.plot_target_by_segment(segment_column=segment_column, plot_type=plot_type)

    # Plot statistical significance
    st.subheader("Plot Statistical Significance")
    ab_test.plot_statistical_significance(test_type=test_type)

    # Check A/B group bias
    st.subheader("Check A/B Group Bias")
    bias_check_column = st.selectbox("Select a column to check for potential bias", data.columns)
    ab_test.check_ab_group_bias(bias_check_column)

    # Plot cumulative metrics
    st.subheader("Plot Cumulative Metrics")
    metric_column = st.selectbox("Select a metric column", data.columns)
    ab_test.plot_cumulative_metrics(metric_column)

    # Plot metric distributions
    st.subheader("Plot Metric Distributions")
    metric = st.selectbox("Select a metric columns", data.columns)
    ab_test.plot_metric_distributions(metric)

    # Analyze seasonality
    st.subheader("Analyze Seasonality")
    ab_test.analyze_seasonality()

    # Plot cumulative target
    st.subheader("Plot Cumulative Target")
    test_type = st.selectbox("Select test type for cumulative target", options=["proportional", "mean"])
    ab_test.plot_cumulative_target(test_type=test_type)

