import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

class HypothesisTester:
    def __init__(self, data):
        self.data = data.copy()
        self.metrics_calculated = False  # Track if KPIs are ready

    # METRICS / KPI CALCULATION

    def calculate_metrics(self):
        """Create KPIs: ClaimOccurred, Claim Frequency, Claim Severity, Margin"""

        # KPI columns
        self.data['ClaimOccurred'] = (self.data['TotalClaims'] > 0).astype(int)
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']

        # Overall KPIs
        claim_frequency = self.data['ClaimOccurred'].mean()
        claim_severity = self.data.loc[self.data['ClaimOccurred'] == 1, 'TotalClaims'].mean() if self.data['ClaimOccurred'].sum() else 0
        margin = self.data['Margin']
        margin_mean = margin.mean()

        self.metrics_calculated = True

        print(f"Metrics calculated: Claim Frequency={claim_frequency:.4f}, Claim Severity={claim_severity:.2f}, Average Margin={margin_mean:.2f}")

        return {"Claim Frequency": claim_frequency, "Claim Severity": claim_severity, "Average Margin": margin_mean}


    # INTERNAL STAT TESTS
    def _chi_square(self, feature, data=None):
        data = data if data is not None else self.data
        table = pd.crosstab(data[feature], data['ClaimOccurred'])
        chi2, p, _, _ = chi2_contingency(table)
        return {"Test": "Chi-Square (Claim Frequency)",
                "Feature": feature,
                "P-Value": p,
                "Reject_H0": p < 0.05}

    def _t_test(self, feature, value_col, data=None):
        data = data if data is not None else self.data
        groups = data[feature].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"T-test requires exactly 2 groups in '{feature}', found {len(groups)}")
        g1 = data[data[feature] == groups[0]][value_col]
        g2 = data[data[feature] == groups[1]][value_col]
        stat, p = ttest_ind(g1, g2, equal_var=False)
        return {"Test": f"T-Test ({value_col})",
                "Feature": feature,
                "Groups": groups.tolist(),
                "Group_1_Mean": g1.mean(),
                "Group_2_Mean": g2.mean(),
                "P-Value": p,
                "Reject_H0": p < 0.05}


    def _interpret(self, result):
        p = result["P-Value"]
        feat = result["Feature"]
        if result["Reject_H0"]:
            result["Interpretation"] = f"Reject H₀ for {feat} (p={p:.4f}) → Statistically significant difference."
        else:
            result["Interpretation"] = f"Fail to reject H₀ for {feat} (p={p:.4f}) → No evidence of difference."
        return result

    def test_province_risk(self):
        if not self.metrics_calculated:
            raise RuntimeError("Calculate metrics first.")
        return self._interpret(self._chi_square("Province"))

    def test_zipcode_risk(self, zip_a, zip_b):
        if not self.metrics_calculated:
            raise RuntimeError("Calculate metrics first.")
        data_zip = self.data[self.data["PostalCode"].isin([zip_a, zip_b])]
        return self._interpret(self._chi_square("PostalCode", data=data_zip))

    def test_zipcode_margin(self, zip_a, zip_b):
        if not self.metrics_calculated:
            raise RuntimeError("Calculate metrics first.")
        data_zip = self.data[self.data["PostalCode"].isin([zip_a, zip_b])]
        return self._interpret(self._t_test("PostalCode", "Margin", data=data_zip))

    def test_gender_risk(self):
        if not self.metrics_calculated:
            raise RuntimeError("Calculate metrics first.")
        data_gender = self.data[self.data["Gender"].isin(["Male", "Female"])]
        freq_result = self._interpret(self._chi_square("Gender", data=data_gender))
        severity_result = self._interpret(self._t_test("Gender", "TotalClaims", data=data_gender))
        return freq_result, severity_result
