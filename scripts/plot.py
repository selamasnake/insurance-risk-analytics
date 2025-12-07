import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDAPlots:
    """
    Visualization utilities for EDA on insurance portfolio data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_segmented_monthly_lr(self, segment_cols):
        """
        Plots segmented Loss Ratios for given columns (e.g., Province, VehicleType, Gender)
        to answer key profitability questions.
        """
        data = self.data
        overall_lr = data['TotalClaims'].sum() / data['TotalPremium'].sum()
        print(f"Overall Loss Ratio: {overall_lr:.2%}")

        for col in segment_cols:
            if col in data.columns:
                segment = data.groupby(col).agg(
                    TotalPremium=('TotalPremium', 'sum'),
                    TotalClaims=('TotalClaims', 'sum')
                )
                segment['LossRatio'] = segment['TotalClaims'] / segment['TotalPremium']
                segment = segment.sort_values('LossRatio', ascending=False)

                plt.figure(figsize=(10, 5))
                sns.barplot(x=segment.index, y=segment['LossRatio'], palette='viridis')
                plt.axhline(overall_lr, color='red', linestyle='--', label='Overall LR')
                plt.title(f"Loss Ratio by {col}")
                plt.xticks(rotation=45)
                plt.legend()
                plt.show()

    def univariate_analysis(self, numerical_cols):
        """
        Plots boxplots and log-scaled histograms for numerical columns
        to detect outliers and visualize distributions.
        """
        data = self.data

        # Boxplots
        plt.figure(figsize=(15, 4))
        for i, col in enumerate(numerical_cols):
            if col in data.columns:
                plt.subplot(1, len(numerical_cols), i + 1)
                sns.boxplot(y=data[col], color='skyblue')
                plt.title(col)
        plt.tight_layout()
        plt.show()

        # Log histograms
        plt.figure(figsize=(15, 4))
        for i, col in enumerate(numerical_cols):
            if col in data.columns:
                plt.subplot(1, len(numerical_cols), i + 1)
                sns.histplot(np.log1p(data[col]), bins=30, kde=True, color='teal')
                plt.title(f'Log({col})')
        plt.tight_layout()
        plt.show()

    def plot_categorical_distributions(self, categorical_cols):
        """
        Plots bar charts for categorical variables to understand value distributions.
        """
        data = self.data
        plt.figure(figsize=(15, 5))
        
        for i, col in enumerate(categorical_cols):
            if col in data.columns:
                plt.subplot(1, len(categorical_cols), i + 1)
                counts = data[col].value_counts()
                sns.barplot(x=counts.index, y=counts.values, palette='pastel')
                plt.xticks(rotation=45)
                plt.title(f"Distribution of {col}")
        
        plt.tight_layout()
        plt.show()

    def plot_monthly_trends(self):
        """
        Plots monthly premiums, claims, and loss ratios over time
        to detect temporal trends in claim frequency and severity.
        """
        data = self.data
        monthly = data.groupby('TransactionMonth').agg(
            TotalPremium=('TotalPremium', 'sum'),
            TotalClaims=('TotalClaims', 'sum'),
            PolicyCount=('PolicyID', 'count')
        )
        monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax1.bar(monthly.index, monthly['PolicyCount'], color='gray', alpha=0.5)
        ax2.plot(monthly.index, monthly['LossRatio'], color='red', marker='o')

        ax1.set_ylabel("Policy Count")
        ax2.set_ylabel("Loss Ratio")
        plt.title("Monthly Policy Count vs Loss Ratio")
        plt.show()

    def plot_vehicle_make_risk(self, top_n=3):
        """
        Plots average premium vs average claim severity for top vehicle makes.
        Bubble size = number of policies.
        Annotates top/bottom N AvgClaim vehicles.
        """
        data = self.data
        make_risk = data.groupby('make').agg(
            AvgPremium=('TotalPremium', 'mean'),
            AvgClaim=('TotalClaims', lambda x: x[x > 0].mean()),
            Volume=('PolicyID', 'count')
        ).dropna()

        high_volume = make_risk[make_risk['Volume'] > make_risk['Volume'].quantile(0.9)]

        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=high_volume,
            x='AvgPremium', y='AvgClaim',
            size='Volume', hue=high_volume.index,
            sizes=(50, 400), legend=False
        )

        # Annotate top N and bottom N AvgClaim
        top_vehicles = high_volume.nlargest(top_n, 'AvgClaim')
        bottom_vehicles = high_volume.nsmallest(top_n, 'AvgClaim')
        for idx, row in pd.concat([top_vehicles, bottom_vehicles]).iterrows():
            plt.text(row['AvgPremium'], row['AvgClaim'], idx, fontsize=9)

        plt.title("Vehicle Make Risk Profile (Annotated Top/Bottom AvgClaim)")
        plt.xlabel("Average Premium")
        plt.ylabel("Average Claim Severity")
        plt.show()

    def plot_zipcode_correlations(self):
        """
        Plots Spearman correlation between monthly premiums and claims for top 5 ZIP codes,
        excluding missing PostalCode values.
        """
        data = self.data
        # Drop NA PostalCodes before counting
        valid_postal_codes = data['PostalCode'].dropna()
        top_zips = valid_postal_codes.value_counts().nlargest(5).index
        correlations = {}

        for z in top_zips:
            zip_data = data[data['PostalCode'] == z]
            monthly = zip_data.groupby('TransactionMonth').agg(
                P=('TotalPremium', 'sum'),
                C=('TotalClaims', 'sum')
            )
            if len(monthly) > 1:
                correlations[z] = monthly['P'].corr(monthly['C'], method='spearman')

        corr_series = pd.Series(correlations).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=corr_series.index, y=corr_series.values, palette='coolwarm')
        plt.ylabel("Spearman Correlation (Premium vs Claims)")
        plt.title("Top 5 ZIP Codes Correlation")
        plt.ylim(-1, 1)
        plt.show()


    def plot_zipcode_scatter(self, top_n=5):
        """
        Plots scatter plots of monthly TotalPremium vs TotalClaims for the top N ZIP codes.
        """
        data = self.data
        top_zips = data['PostalCode'].value_counts().nlargest(top_n).index
        
        plt.figure(figsize=(12, 6))
        
        for zip_code in top_zips:
            zip_data = data[data['PostalCode'] == zip_code].groupby('TransactionMonth').agg(
                TotalPremium=('TotalPremium', 'sum'),
                TotalClaims=('TotalClaims', 'sum')
            )
            sns.scatterplot(
                x=zip_data['TotalPremium'],
                y=zip_data['TotalClaims'],
                label=f"ZIP {zip_code}",
                s=70
            )
            sns.regplot(
                x=zip_data['TotalPremium'], y=zip_data['TotalClaims'],
                scatter=False, ci=None
            )
        
        plt.xlabel("Monthly Total Premium")
        plt.ylabel("Monthly Total Claims")
        plt.title("Monthly Premium vs Claims by ZIP Code")
        plt.legend()
        plt.show()

    def plot_custom_value_distribution(self):
        """
        Plots histogram and boxplot for CustomValueEstimate to visualize outliers.
        """
        data = self.data
        if 'CustomValueEstimate' not in data.columns:
            print("CustomValueEstimate not found in dataset.")
            return

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=data['CustomValueEstimate'], color='lightcoral')
        plt.title("CustomValueEstimate - Outliers")

        plt.subplot(1, 2, 2)
        sns.histplot(data['CustomValueEstimate'].dropna(), bins=30, kde=True, color='coral')
        plt.title("CustomValueEstimate - Distribution")

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, numerical_cols=None):
        """
        Plots a correlation heatmap for numerical variables to understand multivariate relationships.
        """
        data = self.data
        if numerical_cols is None:
            numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
        corr = data[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Correlation Heatmap")
        plt.show()
