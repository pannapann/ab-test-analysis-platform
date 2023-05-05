import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower, NormalIndPower, GofChisquarePower, tt_solve_power
import scipy.stats as stats
import statsmodels.stats.api as sms
import scipy.stats as stats
from plotly.offline import plot
import statsmodels.tsa.api as smt
from plotly.subplots import make_subplots
import streamlit as st


class ABtestTrackingClass:
    """
    This class provides functionality for managing and analyzing A/B test data.
    """
    
    def __init__(self, dataframe, target_column, date_column, group_column):
        """
        Initialize an ABtestTrackingClass object.

        Args:
            dataframe (DataFrame): A pandas DataFrame or Spark DataFrame containing the data for the A/B test.
            target_column (str): The name of the target column in the DataFrame.
            date_column (str): The name of the date column in the DataFrame.
            group_column (str): The name of the group column in the DataFrame.
        """
        

        self.dataframe = dataframe
        self.target_column = target_column
        self.date_column = date_column
        self.group_column = group_column
        
        # if len(self.dataframe[self.target_column].unique())==2:
        #     self.dataframe[self.target_column] = self.dataframe[self.target_column].astype(object)

        # Check if the target_column has exactly two unique values
        unique_targets = self.dataframe[self.target_column].unique()
        if len(unique_targets) == 2:
            # Check if the unique values are 0 and 1
            sorted_targets = sorted(unique_targets)
            if sorted_targets == [0, 1]:
                self.dataframe[self.target_column] = self.dataframe[self.target_column].astype(int)
            elif sorted_targets == ['0', '1']:
                self.dataframe[self.target_column] = self.dataframe[self.target_column].astype(int)
            else:
                raise ValueError('The target_column must have integer values 0 and 1 for proportional data. Please convert the target column values to 0 and 1 before using this platform.')
            
        # Convert the date_column to datetime
        self.dataframe[self.date_column] = pd.to_datetime(self.dataframe[self.date_column], infer_datetime_format=True, errors='coerce')
        self.dataframe[self.date_column] = self.dataframe[self.date_column].dt.strftime('%d/%m/%Y')



        # Check if the group_column has exactly two unique values
        unique_groups = self.dataframe[self.group_column].unique()
        if len(unique_groups) != 2:
            raise ValueError('The group_column must have exactly two unique values.')

        self.group_a, self.group_b = unique_groups
            

    def calculate_effect_size(self, test_type):
        """
        Calculate the effect size for the given test type.

        Args:
            test_type (str): The type of the test ('t_test', 'proportional_z', 'chi_square', or 'mann_whitney_u').

        Returns:
            float: The effect size for the given test type.
        """
        
        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]

        if test_type == 't_test':
            cohen_d = (group_a.mean() - group_b.mean()) / (np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2))
            effect_size = cohen_d
        elif test_type == 'proportional_z':
            p1 = group_a.sum() / group_a.count()  # proportion for control group
            p2 = group_b.sum() / group_b.count()  # proportion for treatment group
            effect_size = sms.proportion_effectsize(p1, p2)
        elif test_type == 'chi_square':
            contingency_table = pd.crosstab(self.dataframe[self.group_column], self.dataframe[self.target_column])
            chi_square, p_value, dof, expected = chi2_contingency(contingency_table)

            n = contingency_table.sum().sum()
            effect_size = np.sqrt(chi_square / (n * (min(contingency_table.shape) - 1)))

        elif test_type == 'mann_whitney_u':
            u_statistic, p_value = stats.mannwhitneyu(group_a, group_b)
            n1 = len(group_a)
            n2 = len(group_b)
            effect_size = abs(u_statistic - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        else:
            raise ValueError('Invalid test_type provided.')
        return effect_size

    def calculate_required_sample_size(self, test_type='t_test', effect_size=None, power=0.8, alpha=0.05):
        """
        Calculate the required sample size for the given test type and effect size.

        Args:
            test_type (str): The type of the test ('t_test', 'proportional_z', 'chi_square', or 'mann_whitney_u').
            effect_size (float, optional): The effect size for the given test type. Defaults to None.
            power (float, optional): The desired power of the test. Defaults to 0.8.
            alpha (float, optional): The desired significance level of the test. Defaults to 0.05.

        Returns:
            float: The required sample size for the given test type and effect size.
        """
        
        if effect_size is None:         
            effect_size = self.calculate_effect_size(test_type)

        if test_type == 't_test':
            power_analysis = TTestIndPower()
            required_sample_size = power_analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
        elif test_type == 'proportional_z':
            power_analysis = NormalIndPower()
            required_sample_size = power_analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
        elif test_type == 'chi_square':
            power_analysis = GofChisquarePower()
            required_sample_size = power_analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
        elif test_type == 'mann_whitney_u':
            raise NotImplementedError('Sample size calculation for Mann-Whitney U test is not supported in this function.')
        else:
            raise ValueError('Invalid test_type provided.')

        return required_sample_size

    def estimate_days_to_collect_data(self, test_type='t_test', effect_size=None, power=0.8, alpha=0.05):
        """
        Estimate the number of days needed to collect the required sample size for the given test type and effect size.

        Args:
            test_type (str): The type of the test ('t_test', 'proportional_z', 'chi_square', or 'mann_whitney_u').
            effect_size (float, optional): The effect size for the given test type. Defaults to None.
            power (float, optional): The desired power of the test. Defaults to 0.8.
            alpha (float, optional): The desired significance level of the test. Defaults to 0.05.

        Returns:
            float: The number of days needed to collect the required sample size.
        """
        
        required_sample_size = int(self.calculate_required_sample_size(test_type, effect_size, power, alpha))
        current_sample_size = int(len(self.dataframe))

        st.write(f"Current sample size: {current_sample_size}")
        st.write(f"Required sample size: {required_sample_size}")
        if current_sample_size >= required_sample_size:
            st.write("The current sample size is enough.")
            return 0
        else:
            days_collected = (self.dataframe[self.date_column].max() - self.dataframe[self.date_column].min()).days
            avg_samples_per_day = current_sample_size / days_collected
            days_needed = (required_sample_size - current_sample_size) / avg_samples_per_day

            st.write(f"Additional {days_needed} days are needed to collect the required sample size.")
            return days_needed

    def select_appropriate_test(self):
        """
        Select the appropriate statistical test based on the characteristics of the data.

        Returns:
            str: The selected test type ('t_test', 'proportional_z', 'chi_square', or 'mann_whitney_u').
        """
        
        # Check if the target column is numeric or categorical
        if (np.issubdtype(self.dataframe[self.target_column].dtype, np.number)) and (self.dataframe[self.target_column].nunique() != 2):

            # If numeric, check if the data is normally distributed
            group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
            group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]

            _, p_value_a = stats.normaltest(group_a)
            _, p_value_b = stats.normaltest(group_b)

            alpha = 0.05
            if p_value_a > alpha and p_value_b > alpha:
                # If both groups are normally distributed, use T-test
                return 't_test'
            else:
                # If not normally distributed, use Mann-Whitney U test
                return 'mann_whitney_u'
        else:
            # If categorical, check if the categories are binary
            unique_categories = self.dataframe[self.target_column].nunique()
            sample_size = len(self.dataframe)

            # Define a sample size threshold
            sample_size_threshold = 50
            
            if unique_categories == 2:
                # If binary, use Proportional Z-test for large sample sizes and Chi-square test for small sample sizes
                if sample_size >= sample_size_threshold:
                    return 'proportional_z'
                else:
                    return 'chi_square'
            else:
                # If not binary, use Chi-square test
                return 'chi_square'
        
    def frequentist_proportion_based(self, test_type=None):
        """
        Perform a frequentist proportion-based statistical test.
        
        Proportional Z-test: The Proportional Z-test is used to compare the proportions of two groups when the following conditions are met:

            The data is categorical (binary) with only two categories.
            The sample size is large enough (usually n >= 30 for each group).
            The samples are independent.
            
        Chi-square test: The Chi-square test is used to analyze the relationship between two categorical variables. It is used when:

            The data is categorical (nominal or ordinal).
            The sample size is large enough (all expected frequencies should be greater than or equal to 5).
            The observations are independent.

        Args:
            test_type (str, optional): The type of the test ('chi_square' or 'proportional_z'). Defaults to None.

        Returns:
            Tuple[float, float]: The test statistic and p-value for the selected test.
        """
        
        success_df = self.dataframe.groupby(self.group_column)[self.target_column].agg(['count', 'mean', 'sum'])
        success_df.columns = [f'total_count', 'success_rate', 'success']
        success_df['non_success'] = success_df['total_count'] - success_df['success']
        st.write(success_df)
        
        if test_type == None:
            test_type = self.select_appropriate_test()
        if test_type == 'chi_square':
            return self.chi_square_test()
        elif test_type == 'proportional_z':
            return self.proportional_z_test()
        else:
            raise ValueError('Invalid test_type provided.')

    def chi_square_test(self):
        """
        Perform a chi-square test.

        Returns:
            str: A summary of the chi-square test results.
        """

        contingency_table = pd.crosstab(self.dataframe[self.group_column], self.dataframe[self.target_column])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        summary = f"Chi-square test:\nChi-square statistic: {chi2:.4f}\nP-value: {p_value:.4f}\n"
        if p_value < 0.05:
            summary += "The p-value is less than 0.05, suggesting that there is a significant difference between the groups."
        else:
            summary += "The p-value is greater than 0.05, suggesting that there is no significant difference between the groups."
        
        self.plot_statistical_significance(test_type='chi_square')

        return summary
    
    def proportional_z_test(self):
        """
        Perform a proportional Z-test.

        Returns:
            str: A summary of the proportional Z-test results.
        """

        group_data = self.dataframe.groupby(self.group_column)[self.target_column].agg(['count', 'sum'])
        count = group_data['count'].values
        converted = group_data['sum'].values
        z_stat, p_value = proportions_ztest(converted, count)

        summary = f"Proportional Z-test:\nZ-test statistic: {z_stat:.4f}\nP-value: {p_value:.4f}\n"
        if p_value < 0.05:
            summary += "The p-value is less than 0.05, suggesting that there is a significant difference between the proportions."
        else:
            summary += "The p-value is greater than 0.05, suggesting that there is no significant difference between the proportions."

        self.plot_statistical_significance(test_type='proportional_z')

        return summary

    def frequentist_mean_based(self, test_type=None):
        """
        Perform a frequentist mean-based statistical test.
        
        T-test: The T-test is used to compare the means of two groups when the following conditions are met:

            The data is continuous (interval or ratio scale).
            The data is approximately normally distributed.
            The variances of the two groups are equal (or close enough).
            The samples are independent.
            
        Mann-Whitney U test: The Mann-Whitney U test is a non-parametric alternative to the T-test, used when the data does not meet the assumptions of the T-test. It is used when:

            The data is continuous, but not normally distributed.
            The variances of the two groups may not be equal.
            The samples are independent.

        Args:
            test_type (str, optional): The type of the test ('t_test' or 'mann_whitney_u'). Defaults to None.

        Returns:
            Tuple[float, float]: The test statistic and p-value for the selected test.
        """
    
        success_df = self.dataframe.groupby(self.group_column)[self.target_column].agg(['mean'])
        success_df.columns = ['mean']
        st.write(success_df)

        if test_type == None:
            test_type = self.select_appropriate_test()
        if test_type == 't_test':
            return self.t_test()
        elif test_type == 'mann_whitney_u_test':
            return self.mann_whitney_u_test()
        else:
            raise ValueError('Invalid test_type provided.')
    
    def t_test(self):
        """
        Perform an independent two-sample t-test between the two groups (group_a and group_b) on the target column.

        Returns:
            str: A summary of the t-test results.
        """

        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]
        t_stat, p_value = ttest_ind(group_a, group_b)

        summary = f"T-test:\nT-test statistic: {t_stat:.4f}\nP-value: {p_value:.4f}\n"
        if p_value < 0.05:
            summary += "The p-value is less than 0.05, suggesting that there is a significant difference between the means."
        else:
            summary += "The p-value is greater than 0.05, suggesting that there is no significant difference between the means."

        self.plot_statistical_significance(test_type='t_test')

        return summary
    
    def mann_whitney_u_test(self):
        """
        Perform a Mann-Whitney U test between the two groups (group_a and group_b) on the target column.

        Returns:
            str: A summary of the Mann-Whitney U test results.
        """

        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]
        u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

        summary = f"Mann-Whitney U test:\nU-test statistic: {u_stat:.4f}\nP-value: {p_value:.4f}\n"
        if p_value < 0.05:
            summary += "The p-value is less than 0.05, suggesting that there is a significant difference between the distributions."
        else:
            summary += "The p-value is greater than 0.05, suggesting that there is no significant difference between the distributions."

        self.plot_statistical_significance(test_type='mann_whitney_u_test')

        return summary
    
    def bayesian_proportion_based(self, n_samples=10000, alpha_prior=1, beta_prior=1):
        """
        Perform a Bayesian analysis using beta distributions for proportion-based data.

        Args:
            n_samples (int, optional): Number of samples to draw from the posterior distributions. Defaults to 10000.
            alpha_prior (int, optional): Pior alpha value
            beta_prior (int, optional): Pior beta value

        Returns:
            dict: A dictionary containing the posterior samples for groups A and B, the probability that group B is better than group A, and the credible intervals for both groups.
        """
        
        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]
        
        success_df = self.dataframe.groupby(self.group_column)[self.target_column].agg(['count', 'mean', 'sum'])
        success_df.columns = [f'total_count', 'success_rate', 'success']
        success_df['non_success'] = success_df['total_count'] - success_df['success']
        st.write(success_df)

        conversions_a = group_a.sum()
        conversions_b = group_b.sum()

        total_a = len(group_a)
        total_b = len(group_b)

        # Calculate posterior parameters
        alpha_a = alpha_prior + conversions_a
        beta_a = beta_prior + total_a - conversions_a
        alpha_b = alpha_prior + conversions_b
        beta_b = beta_prior + total_b - conversions_b

        # Sample from posterior distributions
        posterior_a = np.random.beta(alpha_a, beta_a, n_samples)
        posterior_b = np.random.beta(alpha_b, beta_b, n_samples)

        # Calculate the probability that group B is better than group A
        prob_b_better = (posterior_b > posterior_a).mean()

        # Calculate credible intervals
        ci_a = np.percentile(posterior_a, [2.5, 97.5])
        ci_b = np.percentile(posterior_b, [2.5, 97.5])

        st.write(f"Probability {self.group_b} is better than {self.group_a}: {prob_b_better * 100:.2f}%")
        st.write(f"95% credible interval for group {self.group_a}: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        st.write(f"95% credible interval for group {self.group_b}: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

        # Calculate the relative performance of group B over group A
        relative_performance = posterior_b / posterior_a

        # Plot the histogram of the relative performance using Plotly
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=relative_performance, nbinsx=50, histnorm='probability', name='Histogram', yaxis='y1'))

        # Calculate and plot the CDF of the relative performance
        cdf_values, cdf_indices = np.histogram(relative_performance, bins=50, density=True)
        cdf_values = cdf_values.cumsum() / cdf_values.sum()
        fig.add_trace(go.Scatter(x=cdf_indices[1:], y=cdf_values, mode='lines', name='CDF', yaxis='y2'))

        # Update layout with secondary y-axis
        fig.update_layout(
            title=f'Relative Performance of {self.group_b} over {self.group_a}',
            xaxis_title=f'Relative Performance ({self.group_b} / {self.group_a})',
            yaxis_title='Probability',
            yaxis=dict(title='Probability', side='left', anchor='x'),
            yaxis2=dict(title='Cumulative Probability', side='right', overlaying='y', anchor='x'),
            legend=dict(x=1, y=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate summary statistics
        median_relative_performance = np.median(relative_performance)
        mean_relative_performance = np.mean(relative_performance)
        pi_relative_performance = np.percentile(relative_performance, [2.5, 97.5])

        st.write(f"Median relative performance of {self.group_b} over {self.group_a}: {median_relative_performance:.2f}")
        st.write(f"Mean relative performance of {self.group_b} over {self.group_a}: {mean_relative_performance:.2f}")
        st.write(f"95% probability interval of relative performance: [{pi_relative_performance[0]:.2f}, {pi_relative_performance[1]:.2f}]")

        return {
            'posterior_samples_a': posterior_a,
            'posterior_samples_b': posterior_b,
            'probability_b_better': prob_b_better,
            'credible_interval_a': ci_a,
            'credible_interval_b': ci_b,
            'relative_performance': relative_performance
        }

    def bayesian_mean_based(self, n_samples=10000):
        """
        Perform a Bayesian analysis using normal distributions for mean-based data.

        Args:
            n_samples (int, optional): Number of samples to draw from the posterior distributions. Defaults to 10000.

        Returns:
            dict: A dictionary containing the posterior samples for groups A and B, the probability that group B is better than group A, and the credible intervals for both groups.
        """
        
        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]

        success_df = self.dataframe.groupby(self.group_column)[self.target_column].agg(['mean'])
        success_df.columns = ['mean']
        st.write(success_df)

        # Calculate the means and standard deviations of the groups
        mean_a, std_a = group_a.mean(), group_a.std()
        mean_b, std_b = group_b.mean(), group_b.std()

        # Sample from the normal distributions
        samples_a = np.random.normal(loc=mean_a, scale=std_a, size=n_samples)
        samples_b = np.random.normal(loc=mean_b, scale=std_b, size=n_samples)

        # Calculate the probability that group B is better than group A
        prob_b_better = (samples_b > samples_a).mean()

        # Calculate the 95% credible intervals for both groups
        ci_a = np.percentile(samples_a, [2.5, 97.5])
        ci_b = np.percentile(samples_b, [2.5, 97.5])

        st.write(f"Probability {self.group_b} is better than {self.group_a}: {prob_b_better * 100:.2f}%")
        st.write(f"95% credible interval for {self.group_a}: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        st.write(f"95% credible interval for {self.group_b}: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

        # Calculate the relative performance of group B over group A
        relative_performance = samples_b / samples_a

        # Plot the histogram of the relative performance using Plotly
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=relative_performance, nbinsx=50, histnorm='probability', name='Histogram', yaxis='y1'))

        # Calculate and plot the CDF of the relative performance
        cdf_values, cdf_indices = np.histogram(relative_performance, bins=50, density=True)
        cdf_values = cdf_values.cumsum() / cdf_values.sum()
        fig.add_trace(go.Scatter(x=cdf_indices[1:], y=cdf_values, mode='lines', name='CDF', yaxis='y2'))

        # Update layout with secondary y-axis
        fig.update_layout(
            title=f'Relative Performance of {self.group_b} over {self.group_a}',
            xaxis_title=f'Relative Performance ({self.group_b} / {self.group_a})',
            yaxis_title='Probability',
            yaxis=dict(title='Probability', side='left', anchor='x'),
            yaxis2=dict(title='Cumulative Probability', side='right', overlaying='y', anchor='x'),
            legend=dict(x=1, y=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate summary statistics
        median_relative_performance = np.median(relative_performance)
        mean_relative_performance = np.mean(relative_performance)
        pi_relative_performance = np.percentile(relative_performance, [2.5, 97.5])

        st.write(f"Median relative performance of {self.group_b} over {self.group_a}: {median_relative_performance:.2f}")
        st.write(f"Mean relative performance of {self.group_b} over {self.group_a}: {mean_relative_performance:.2f}")
        st.write(f"95% probability interval of relative performance: [{pi_relative_performance[0]:.2f}, {pi_relative_performance[1]:.2f}]")

        return {
            'posterior_samples_a': samples_a,
            'posterior_samples_b': samples_b,
            'probability_b_better': prob_b_better,
            'credible_interval_a': ci_a,
            'credible_interval_b': ci_b,
            'relative_performance': relative_performance
        }

    def plot_target_by_date(self, aggregation='day', plot_type='proportion_based'):
        """
        Plot the target variable aggregated by date for each group (group_a and group_b).

        Args:
            aggregation (str, optional): Time aggregation level ('day', 'month', or 'year'). Defaults to 'day'.
            plot_type (str, optional): Type of plot to generate ('proportion_based' or 'mean_based'). Defaults to 'proportion_based'.
        """
        
        df = self.dataframe.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column],infer_datetime_format=True)

        if aggregation == 'month':
            df[self.date_column] = df[self.date_column].dt.to_period('M')
        elif aggregation == 'year':
            df[self.date_column] = df[self.date_column].dt.to_period('Y')

        if plot_type == 'proportion_based':
            summary = df.groupby([self.date_column, self.group_column])[self.target_column].agg(['count', 'sum']).reset_index()
            summary['proportion'] = summary['sum'] / summary['count']
            fig = px.line(summary, x=self.date_column, y='proportion', color=self.group_column, title='Target Proportion by Date')
        elif plot_type == 'mean_based':
            summary = df.groupby([self.date_column, self.group_column])[self.target_column].mean().reset_index()
            fig = px.line(summary, x=self.date_column, y=self.target_column, color=self.group_column, title='Target Mean by Date')
        else:
            raise ValueError('Invalid plot_type provided.')

        st.plotly_chart(fig, use_container_width=True)

    def plot_target_by_segment(self, segment_column, plot_type='proportion_based', plot_format='bar'):
        """
        Plot the target variable aggregated by a specified segment column for each group (group_a and group_b).

        Args:
            segment_column (str): Column to segment the data by.
            plot_type (str, optional): Type of plot to generate ('proportion_based' or 'mean_based'). Defaults to 'proportion_based'.
            plot_format (str, optional): Format of the plot ('bar' or 'heatmap'). Defaults to 'bar'.
        """
        
        if plot_type == 'proportion_based':
            summary = self.dataframe.groupby([segment_column, self.group_column])[self.target_column].agg(['count', 'sum']).reset_index()
            summary['proportion'] = summary['sum'] / summary['count']
            if plot_format == 'bar':
                fig = px.bar(summary, x=segment_column, y='proportion', color=self.group_column, title='Target Proportion by Segment', barmode='group')
            elif plot_format == 'heatmap':
                fig = px.density_heatmap(summary, x=segment_column, y=self.group_column, z='proportion', title='Target Proportion by Segment')
            else:
                raise ValueError('Invalid plot_format provided.')

        elif plot_type == 'mean_based':
            summary = self.dataframe.groupby([segment_column, self.group_column])[self.target_column].mean().reset_index()
            if plot_format == 'bar':
                fig = px.bar(summary, x=segment_column, y=self.target_column, color=self.group_column, title='Target Mean by Segment', barmode='group')
            elif plot_format == 'heatmap':
                fig = px.density_heatmap(summary, x=segment_column, y=self.group_column, z=self.target_column, title='Target Mean by Segment')
            else:
                raise ValueError('Invalid plot_format provided.')

        else:
            raise ValueError('Invalid plot_type provided.')

        st.plotly_chart(fig, use_container_width=True)
    
    def bayesian_proportion_based(self, n_samples=10000, alpha_prior=1, beta_prior=1):
        """
        Perform a Bayesian analysis using beta distributions for proportion-based data.

        Args:
            n_samples (int, optional): Number of samples to draw from the posterior distributions. Defaults to 10000.
            alpha_prior (int, optional): Pior alpha value
            beta_prior (int, optional): Pior beta value

        Returns:
            dict: A dictionary containing the posterior samples for groups A and B, the probability that group B is better than group A, and the credible intervals for both groups.
        """
        
        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]
        
        success_df = self.dataframe.groupby(self.group_column)[self.target_column].agg(['count', 'mean', 'sum'])
        success_df.columns = [f'total_count', 'success_rate', 'success']
        success_df['non_success'] = success_df['total_count'] - success_df['success']
        st.write(success_df)

        conversions_a = group_a.sum()
        conversions_b = group_b.sum()

        total_a = len(group_a)
        total_b = len(group_b)

        # Calculate posterior parameters
        alpha_a = alpha_prior + conversions_a
        beta_a = beta_prior + total_a - conversions_a
        alpha_b = alpha_prior + conversions_b
        beta_b = beta_prior + total_b - conversions_b

        # Sample from posterior distributions
        posterior_a = np.random.beta(alpha_a, beta_a, n_samples)
        posterior_b = np.random.beta(alpha_b, beta_b, n_samples)

        # Calculate the probability that group B is better than group A
        prob_b_better = (posterior_b > posterior_a).mean()

        # Calculate credible intervals
        ci_a = np.percentile(posterior_a, [2.5, 97.5])
        ci_b = np.percentile(posterior_b, [2.5, 97.5])

        st.write(f"Probability {self.group_b} is better than {self.group_a}: {prob_b_better * 100:.2f}%")
        st.write(f"95% credible interval for group {self.group_a}: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        st.write(f"95% credible interval for group {self.group_b}: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

        # Calculate the relative performance of group B over group A
        relative_performance = posterior_b / posterior_a

        # Plot the histogram of the relative performance using Plotly
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=relative_performance, nbinsx=50, histnorm='probability', name='Histogram', yaxis='y1'))

        # Calculate and plot the CDF of the relative performance
        cdf_values, cdf_indices = np.histogram(relative_performance, bins=50, density=True)
        cdf_values = cdf_values.cumsum() / cdf_values.sum()
        fig.add_trace(go.Scatter(x=cdf_indices[1:], y=cdf_values, mode='lines', name='CDF', yaxis='y2'))

        # Update layout with secondary y-axis
        fig.update_layout(
            title=f'Relative Performance of {self.group_b} over {self.group_a}',
            xaxis_title=f'Relative Performance ({self.group_b} / {self.group_a})',
            yaxis_title='Probability',
            yaxis=dict(title='Probability', side='left', anchor='x'),
            yaxis2=dict(title='Cumulative Probability', side='right', overlaying='y', anchor='x'),
            legend=dict(x=1, y=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate summary statistics
        median_relative_performance = np.median(relative_performance)
        mean_relative_performance = np.mean(relative_performance)
        pi_relative_performance = np.percentile(relative_performance, [2.5, 97.5])

        st.write(f"Median relative performance of {self.group_b} over {self.group_a}: {median_relative_performance:.2f}")
        st.write(f"Mean relative performance of {self.group_b} over {self.group_a}: {mean_relative_performance:.2f}")
        st.write(f"95% probability interval of relative performance: [{pi_relative_performance[0]:.2f}, {pi_relative_performance[1]:.2f}]")

        return {
            'posterior_samples_a': posterior_a,
            'posterior_samples_b': posterior_b,
            'probability_b_better': prob_b_better,
            'credible_interval_a': ci_a,
            'credible_interval_b': ci_b,
            'relative_performance': relative_performance
        }

    def bayesian_mean_based(self, n_samples=10000):
        """
        Perform a Bayesian analysis using normal distributions for mean-based data.

        Args:
            n_samples (int, optional): Number of samples to draw from the posterior distributions. Defaults to 10000.

        Returns:
            dict: A dictionary containing the posterior samples for groups A and B, the probability that group B is better than group A, and the credible intervals for both groups.
        """
        
        group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][self.target_column]
        group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][self.target_column]

        success_df = self.dataframe.groupby(self.group_column)[self.target_column].agg(['mean'])
        success_df.columns = ['mean']
        st.write(success_df)

        # Calculate the means and standard deviations of the groups
        mean_a, std_a = group_a.mean(), group_a.std()
        mean_b, std_b = group_b.mean(), group_b.std()

        # Sample from the normal distributions
        samples_a = np.random.normal(loc=mean_a, scale=std_a, size=n_samples)
        samples_b = np.random.normal(loc=mean_b, scale=std_b, size=n_samples)

        # Calculate the probability that group B is better than group A
        prob_b_better = (samples_b > samples_a).mean()

        # Calculate the 95% credible intervals for both groups
        ci_a = np.percentile(samples_a, [2.5, 97.5])
        ci_b = np.percentile(samples_b, [2.5, 97.5])

        st.write(f"Probability {self.group_b} is better than {self.group_a}: {prob_b_better * 100:.2f}%")
        st.write(f"95% credible interval for {self.group_a}: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        st.write(f"95% credible interval for {self.group_b}: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

        # Calculate the relative performance of group B over group A
        relative_performance = samples_b / samples_a

        # Plot the histogram of the relative performance using Plotly
        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(x=relative_performance, nbinsx=50, histnorm='probability', name='Histogram', yaxis='y1'))

        # Calculate and plot the CDF of the relative performance
        cdf_values, cdf_indices = np.histogram(relative_performance, bins=50, density=True)
        cdf_values = cdf_values.cumsum() / cdf_values.sum()
        fig.add_trace(go.Scatter(x=cdf_indices[1:], y=cdf_values, mode='lines', name='CDF', yaxis='y2'))

        # Update layout with secondary y-axis
        fig.update_layout(
            title=f'Relative Performance of {self.group_b} over {self.group_a}',
            xaxis_title=f'Relative Performance ({self.group_b} / {self.group_a})',
            yaxis_title='Probability',
            yaxis=dict(title='Probability', side='left', anchor='x'),
            yaxis2=dict(title='Cumulative Probability', side='right', overlaying='y', anchor='x'),
            legend=dict(x=1, y=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate summary statistics
        median_relative_performance = np.median(relative_performance)
        mean_relative_performance = np.mean(relative_performance)
        pi_relative_performance = np.percentile(relative_performance, [2.5, 97.5])

        st.write(f"Median relative performance of {self.group_b} over {self.group_a}: {median_relative_performance:.2f}")
        st.write(f"Mean relative performance of {self.group_b} over {self.group_a}: {mean_relative_performance:.2f}")
        st.write(f"95% probability interval of relative performance: [{pi_relative_performance[0]:.2f}, {pi_relative_performance[1]:.2f}]")

        return {
            'posterior_samples_a': samples_a,
            'posterior_samples_b': samples_b,
            'probability_b_better': prob_b_better,
            'credible_interval_a': ci_a,
            'credible_interval_b': ci_b,
            'relative_performance': relative_performance
        }

    def plot_target_by_date(self, aggregation='day', plot_type='proportion_based'):
        """
        Plot the target variable aggregated by date for each group (group_a and group_b).

        Args:
            aggregation (str, optional): Time aggregation level ('day', 'month', or 'year'). Defaults to 'day'.
            plot_type (str, optional): Type of plot to generate ('proportion_based' or 'mean_based'). Defaults to 'proportion_based'.
        """
        
        df = self.dataframe.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column],infer_datetime_format=True)

        if aggregation == 'month':
            df[self.date_column] = df[self.date_column].dt.to_period('M')
        elif aggregation == 'year':
            df[self.date_column] = df[self.date_column].dt.to_period('Y')

        if plot_type == 'proportion_based':
            summary = df.groupby([self.date_column, self.group_column])[self.target_column].agg(['count', 'sum']).reset_index()
            summary['proportion'] = summary['sum'] / summary['count']
            fig = px.line(summary, x=self.date_column, y='proportion', color=self.group_column, title='Target Proportion by Date')
        elif plot_type == 'mean_based':
            summary = df.groupby([self.date_column, self.group_column])[self.target_column].mean().reset_index()
            fig = px.line(summary, x=self.date_column, y=self.target_column, color=self.group_column, title='Target Mean by Date')
        else:
            raise ValueError('Invalid plot_type provided.')

        st.plotly_chart(fig, use_container_width=True)

    def plot_target_by_segment(self, segment_column, plot_type='proportion_based', plot_format='bar'):
        """
        Plot the target variable aggregated by a specified segment column for each group (group_a and group_b).

        Args:
            segment_column (str): Column to segment the data by.
            plot_type (str, optional): Type of plot to generate ('proportion_based' or 'mean_based'). Defaults to 'proportion_based'.
            plot_format (str, optional): Format of the plot ('bar' or 'heatmap'). Defaults to 'bar'.
        """
        
        if plot_type == 'proportion_based':
            summary = self.dataframe.groupby([segment_column, self.group_column])[self.target_column].agg(['count', 'sum']).reset_index()
            summary['proportion'] = summary['sum'] / summary['count']
            if plot_format == 'bar':
                fig = px.bar(summary, x=segment_column, y='proportion', color=self.group_column, title='Target Proportion by Segment', barmode='group')
            elif plot_format == 'heatmap':
                fig = px.density_heatmap(summary, x=segment_column, y=self.group_column, z='proportion', title='Target Proportion by Segment')
            else:
                raise ValueError('Invalid plot_format provided.')

        elif plot_type == 'mean_based':
            summary = self.dataframe.groupby([segment_column, self.group_column])[self.target_column].mean().reset_index()
            if plot_format == 'bar':
                fig = px.bar(summary, x=segment_column, y=self.target_column, color=self.group_column, title='Target Mean by Segment', barmode='group')
            elif plot_format == 'heatmap':
                fig = px.density_heatmap(summary, x=segment_column, y=self.group_column, z=self.target_column, title='Target Mean by Segment')
            else:
                raise ValueError('Invalid plot_format provided.')

        else:
            raise ValueError('Invalid plot_type provided.')

        st.plotly_chart(fig, use_container_width=True)

    def plot_statistical_significance(self, test_type='t_test'):
        """
        Plot the cumulative p-values and sample sizes for each day using the specified statistical test.

        Args:
            test_type (str, optional): Type of statistical test to perform ('t_test', 'mann_whitney_u', 'proportional_z', or 'chi_square'). Defaults to 't_test'.
        """
        
        # Prepare data
        df = self.dataframe.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column],infer_datetime_format=True)
        df[self.date_column] = df[self.date_column].dt.to_period('D')
        df = df.sort_values(by=self.date_column)
        unique_dates = df[self.date_column].unique()

        # Calculate cumulative p-values and sample sizes for each day
        p_values = []
        sample_sizes = []
        for date in unique_dates:
            temp_df = df[df[self.date_column] <= date]
            group_a = temp_df[temp_df[self.group_column] == self.group_a][self.target_column]
            group_b = temp_df[temp_df[self.group_column] == self.group_b][self.target_column]

            if test_type == 't_test':
                _, p_value = ttest_ind(group_a, group_b)
            elif test_type == 'mann_whitney_u_test':
                _, p_value = mannwhitneyu(group_a, group_b)
            elif test_type == 'proportional_z':
                count = np.array([group_a.sum(), group_b.sum()])
                nobs = np.array([len(group_a), len(group_b)])
                _, p_value = proportions_ztest(count, nobs)
            elif test_type == 'chi_square':
                contingency_table = pd.crosstab(temp_df[self.group_column], temp_df[self.target_column])
                _, p_value, _, _ = chi2_contingency(contingency_table)
            else:
                raise ValueError('Invalid test_type provided.')

            p_values.append(p_value)
            sample_sizes.append(len(temp_df))

        # Plot p-values and sample sizes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=unique_dates.to_timestamp(), y=sample_sizes, mode='lines+markers', name='Cumulative Sample Size'), secondary_y=False)
        fig.add_trace(go.Scatter(x=unique_dates.to_timestamp(), y=p_values, mode='lines+markers', name='P-value'), secondary_y=True)
        fig.update_layout(title=f'Statistical Significance ({test_type}) by Day', xaxis_title='Date')
        fig.update_yaxes(title_text="Cumulative Sample Size", secondary_y=False)
        fig.update_yaxes(title_text="P-value", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    def check_ab_group_bias(self, column_name):
        """
        Check for potential bias in the distribution of the specified column between the two groups (group_a and group_b).

        Args:
            column_name (str): Name of the column to check for potential bias.
        """
        
        # Check if the column is numeric or categorical
        if np.issubdtype(self.dataframe[column_name].dtype, np.number):
            # Plot histogram for numeric data
            fig = px.histogram(self.dataframe, x=column_name, color=self.group_column, nbins=20, barnorm='fraction', histnorm='percent', barmode='group')
            fig.update_layout(title=f'Distribution of {column_name} by {self.group_a} and {self.group_b} Group', xaxis_title=column_name, yaxis_title='Percentage')

            # Perform t-test
            group_a = self.dataframe[self.dataframe[self.group_column] == self.group_a][column_name]
            group_b = self.dataframe[self.dataframe[self.group_column] == self.group_b][column_name]
            t_stat, p_value = ttest_ind(group_a, group_b)
            test_name = 'T-test'
        else:
            # Plot grouped bar chart for categorical data
            count_df = self.dataframe.groupby([self.group_column, column_name]).size().reset_index(name='count')
            count_df['count'] = count_df['count'] / count_df.groupby(self.group_column)['count'].transform('sum')
            fig = px.bar(count_df, x=column_name, y='count', color=self.group_column, barmode='group')
            fig.update_layout(title=f'Distribution of {column_name} by {self.group_a} and {self.group_b} Group', xaxis_title=column_name, yaxis_title='Density')

            # Perform chi-square test
            contingency_table = pd.crosstab(self.dataframe[self.group_column], self.dataframe[column_name])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            test_name = 'Chi-square test'

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

        # Display basic stats and test results
        control_stats = self.dataframe[self.dataframe[self.group_column] == self.group_a][column_name].describe()
        treatment_stats = self.dataframe[self.dataframe[self.group_column] == self.group_b][column_name].describe()
        st.write(f'Basic stats for {self.group_a} group:\n{control_stats}\n')
        st.write(f'Basic stats for {self.group_b} group:\n{treatment_stats}\n')
        st.write(f'{test_name} result:')
        st.write(f'P-value: {p_value}')
        
        # Interpret the p-value
        alpha = 0.05
        if p_value < alpha:
            st.write(f'The p-value ({p_value:.4f}) is less than the significance level of {alpha}, suggesting there is a statistically significant difference between the groups.')
        else:
            st.write(f'The p-value ({p_value:.4f}) is greater than the significance level of {alpha}, suggesting there is no statistically significant difference between the groups.')
    
    def plot_cumulative_metrics(self, metric_column, colors=None, line_dash=None):
        """
        Plot the cumulative values of the specified metric column by date for each group (group_a and group_b).

        Args:
            metric_column (str): Column containing the metric to plot.
            colors (list, optional): List of colors to use in the plot.
        """
        
        df = self.dataframe.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column],infer_datetime_format=True)
        df = df.sort_values(by=self.date_column)
        
        df_cumsum = df.groupby([self.date_column, self.group_column]).agg({metric_column: 'sum'}).groupby(self.group_column).cumsum().reset_index()
        
        fig = px.line(df_cumsum, x=self.date_column, y=metric_column, color=self.group_column, title=f'Cumulative {metric_column} by {self.group_column}', color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

    def plot_metric_distributions(self, metric_column, colors=None, nbins=50, normalize=False):
        """
        Plot the distribution of the specified metric for both group A and group B using histograms.

        Args:
            metric_column (str): The name of the column containing the metric to be plotted.
            colors (list, optional): A list of colors to use for the histograms.
            nbins (int, optional): The number of bins to use for the histograms (default: 50).
            normalize (bool, optional): Normalize the histograms to show percentage (default: False).
        """

        df = self.dataframe.copy()

        histnorm = 'percent' if normalize else None
        barnorm = 'fraction' if normalize else None

        fig = px.histogram(df, x=metric_column, color=self.group_column, nbins=nbins, title=f'Distribution of {metric_column} by {self.group_column}', color_discrete_sequence=colors, histnorm=histnorm, barnorm=barnorm)
        st.plotly_chart(fig, use_container_width=True)
        
    def analyze_seasonality(self, period=365):
        """
        Analyze the seasonality of the target column in the dataframe and plot the seasonal decomposition using Plotly.

        Trend: This line represents the long-term progression of the time series data. It shows the overall direction of the data, ignoring any seasonal fluctuations.

        Seasonal: This line represents the repeating patterns or fluctuations that occur within each seasonal cycle. It captures the seasonality in the data. 

        Residual: This line represents the remaining variations in the data after the Trend and Seasonal components have been removed. It includes any noise, irregularities, or other unexplained variations. 

        Args:
            period (int, optional): The number of periods in a complete seasonal cycle. Default is 365.
        """
        data = self.dataframe.sort_values(by=self.date_column)
        data.set_index(self.date_column, inplace=True)
        seasonal_decomposition = smt.seasonal_decompose(data[self.target_column], period=period)

        # Plot the seasonal decomposition using Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index, y=seasonal_decomposition.trend, mode='lines', name='Trend'))
        fig.add_trace(go.Scatter(x=data.index, y=seasonal_decomposition.seasonal, mode='lines', name='Seasonal'))
        fig.add_trace(go.Scatter(x=data.index, y=seasonal_decomposition.resid, mode='lines', name='Residual'))

        fig.update_layout(title='Seasonal Decomposition', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)
        
    
    def plot_cumulative_target(self, test_type='proportion_based'):
        """
        Plot the cumulative target (conversion rate or mean) over time for both groups (A and B).

        This function generates a line plot of the cumulative target for each group over time, allowing users to visualize the performance of the test and control groups throughout the A/B test. The target is calculated as the cumulative sum of conversions or values divided by the cumulative count of observations for each group, depending on the test_type parameter.
        """

        if test_type not in ['proportion_based', 'mean_based']:
            raise ValueError("test_type must be either 'proportion_based' or 'mean_based'")

        # Group the data by date and group_column, and calculate the cumulative sum
        cumulative_data = self.dataframe.groupby([self.date_column, self.group_column])[self.target_column].agg(['sum', 'count']).reset_index()
        cumulative_data['cumulative_sum'] = cumulative_data.groupby(self.group_column)['sum'].cumsum()
        cumulative_data['cumulative_count'] = cumulative_data.groupby(self.group_column)['count'].cumsum()
        
        if test_type == 'proportion_based':
            cumulative_data['cumulative_target'] = cumulative_data['cumulative_sum'] / cumulative_data['cumulative_count']
            metric_title = 'Cumulative Conversion Rate'
        else: # test_type == 'mean_based'
            cumulative_data['cumulative_target'] = cumulative_data['cumulative_sum'] / cumulative_data['cumulative_count']
            metric_title = 'Cumulative Mean'

        # Split the data into group A and group B
        group_a_data = cumulative_data[cumulative_data[self.group_column] == self.group_a]
        group_b_data = cumulative_data[cumulative_data[self.group_column] == self.group_b]

        # Plot the cumulative metric for both groups
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group_a_data[self.date_column], y=group_a_data['cumulative_target'], mode='lines', name=f'Group {self.group_a}'))
        fig.add_trace(go.Scatter(x=group_b_data[self.date_column], y=group_b_data['cumulative_target'], mode='lines', name=f'Group {self.group_b}'))

        fig.update_layout(title=f'{metric_title} Over Time', xaxis_title='Date', yaxis_title=metric_title)
        st.plotly_chart(fig, use_container_width=True)