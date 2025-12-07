import pandas as pd

class ExploratoryDataAnalysis:
    """
    class for Exploratory Data Analysis.
    """
    def __init__(self, data):
        self.data = data

    def descriptive_statistics(self, numerical_cols):
        """
        Returns standard data.describe() and a variability table with mean, median, std, and CV.
        """
        desc = self.data[numerical_cols].describe().round(2)
        
        # Variability: mean, median, std
        variability = self.data[numerical_cols].agg(['mean', 'median', 'std'])
        
        # Coefficient of Variation
        cv = (variability.loc['std'] / variability.loc['mean']).to_frame().T
        cv.index = ['CV']
        variability = pd.concat([variability, cv]).round(2)
        
        return desc, variability

    def calculate_loss_ratios(self, segment_cols):
        data = self.data
        results = {}

        overall_lr = data['TotalClaims'].sum() / data['TotalPremium'].sum()
        results['overall'] = overall_lr

        for col in segment_cols:
            if col in data.columns:
                segment = data.groupby(col, observed=True).agg(
                    TotalPremium=('TotalPremium', 'sum'),
                    TotalClaims=('TotalClaims', 'sum')
                )
                segment['LossRatio'] = segment['TotalClaims'] / segment['TotalPremium']
                results[col] = segment

        return results

    def bivariate_analysis(self):
        """
        Produces bivariate/trend-focused summaries:
        - Monthly total premiums, claims, and loss ratios
        - Average claim severity per vehicle make
        - Spearman correlation of monthly premium vs claims in top ZIP codes
        """
        data = self.data

        # Monthly totals
        monthly = data.groupby('TransactionMonth').agg(
            TotalPremium=('TotalPremium', 'sum'),
            TotalClaims=('TotalClaims', 'sum')
        )
        monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']

        # Claim severity per make
        make_severity = (
            data[data['TotalClaims'] > 0]
            .groupby('make')['TotalClaims']
            .mean()
            .sort_values(ascending=False)
        )

        # Correlations by top ZIP codes
        top_zips = data['PostalCode'].value_counts().nlargest(5).index
        correlations = {}

        for z in top_zips:
            zip_data = data[data['PostalCode'] == z]
            m = zip_data.groupby('TransactionMonth').agg(
                P=('TotalPremium', 'sum'),
                C=('TotalClaims', 'sum')
            )
            if len(m) > 1:
                correlations[z] = m['P'].corr(m['C'], method='spearman')

        return {
            "monthly_summary": monthly,
            "make_severity": make_severity,
            "zip_correlations": pd.Series(correlations)
        }