"""
Docstring for scripts.data_preprocessing
"""

import pandas as pd
import numpy as np

class PreprocessData: 
    """
    A class to perform data understanding, type conversion, and basic preprocessing for the insurance dataset.
    """

    def __init__(self, data):
        self.data = data

    def understand_data(self, data):
        """
        Prints the info of the dataframe
        """
        return data.info()

    
    def explore_by_columns(self, data):
        """Grouped columns for easier inspection.
        """
        column_groups = {
        "Columns about the Insurance Policy": [
            "UnderwrittenCoverID", "PolicyID", "TransactionMonth"
        ],

        "Columns about the Client": [
            "IsVATRegistered", "Citizenship", "LegalType", "Title", 
            "Language", "Bank", "AccountType", "MaritalStatus", "Gender"
        ],

        "Columns about the Client location": [
            "Country", "Province", "PostalCode", "MainCrestaZone", "SubCrestaZone"
        ],

        "Columns about the Car Insured": [
            "ItemType", "mmcode", "VehicleType", "RegistrationYear", "make", 
            "Model", "Cylinders", "cubiccapacity", "kilowatts", "bodytype", 
            "NumberOfDoors", "VehicleIntroDate", "CustomValueEstimate", 
            "AlarmImmobiliser", "TrackingDevice", "CapitalOutstanding", 
            "NewVehicle", "WrittenOff", "Rebuilt", "Converted", 
            "CrossBorder", "NumberOfVehiclesInFleet"
        ],

        "Columns about the Plan": [
            "SumInsured", "TermFrequency", "CalculatedPremiumPerTerm", 
            "ExcessSelected", "CoverCategory", "CoverType", "CoverGroup", 
            "Section", "Product", "StatutoryClass", "StatutoryRiskType"
        ],

        "Columns about the Payment & Claim": [
            "TotalPremium", "TotalClaims"
        ]
        }

        # Print grouped heads
        for group_name, cols in column_groups.items():
            print(f"\n{group_name}")
            display(data[cols].tail())

    def convert_data_types(self, data):

        data['PostalCode'] = data['PostalCode'].astype(str)
        data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], format='mixed')
        data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], format='mixed')
        data['CapitalOutstanding'] = pd.to_numeric(data['CapitalOutstanding'], errors='coerce')

        categorical_cols = [
            'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 
            'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Country', 
            'Province', 'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType',
            'make', 'Model', 'bodytype', 'AlarmImmobiliser', 'TrackingDevice', 
            'TermFrequency', 'CoverCategory', 'CoverType', 'CoverGroup', 
            'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType' 'WrittenOff', 'Rebuilt', 'Converted'
        ]
        
        # Filter for columns that actually exist in the DataFrame (to prevent errors)
        cols_to_convert = [col for col in categorical_cols if col in data.columns]

        for col in cols_to_convert:
            # treat the PostalCode (now a string) as a category
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category')
                
        print("--- Updated Dtypes---")
        print(data.info())

    def check_missing_values(self, data):
        """
        Check missing values in each column.
        Returns a DataFrame with missing values count and percentage for each column.
        """
        missing_count = data.isna().sum()
        missing_percent = (missing_count / len(data) * 100).round(2)
        missing_df = pd.DataFrame({
            "MissingCount": missing_count,
            "MissingPercent": missing_percent
        })
        missing_df = missing_df[missing_df["MissingCount"] > 0]

        return missing_df.sort_values(by="MissingPercent", ascending=False)

    def handle_missing_values(self, data):

        # Drop columns with >99% missing values
        drop_cols = ['NumberOfVehiclesInFleet', 'CrossBorder']
        data = data.drop(columns=[col for col in drop_cols if col in data.columns])

        # # Replace 0s in CustomValueEstimate with NaN
        # if 'CustomValueEstimate' in data.columns:
        #     data['CustomValueEstimate'] = data['CustomValueEstimate'].replace(0, np.nan)

        # Fill high-risk vehicle flags with 'No'
        risk_flags = ['WrittenOff', 'Rebuilt', 'Converted']
        for col in risk_flags:
            if col in data.columns:
                data[col] = data[col].fillna('No')

        return data


