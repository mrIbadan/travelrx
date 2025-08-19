#!pip install pycountry

import pandas as pd
import gdown
import io
import tempfile
import os
import numpy as np
import uuid
import hashlib
import pycountry
from faker import Faker
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def google_drive_read(url, **kwargs):
    """
    Download a CSV file from Google Drive and read into pandas.
    
    Parameters:
    - url: str, Google Drive shareable link
    - kwargs: any additional args to pass to pd.read_csv
    
    Returns:
    - pandas DataFrame
    """
    # Extract file ID from URL
    if "/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]
    else:
        raise ValueError("Cannot extract file ID from URL. Check the format.")
    
    # Create direct download URL
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    # Use a temporary file to store the downloaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
    
    # Download content to the temporary file
    try:
        gdown.download(download_url, output=file_path, quiet=False)
        
        # Read the temporary file into a pandas DataFrame
        df = pd.read_csv(file_path, **kwargs)
        
    finally:
        # Clean up by removing the temporary file
        os.remove(file_path)
    
    return df

# Define columns to read
cols = ['pcd', 'country', 'region', 'admin_district']

# Usage
try:
    geo_df = google_drive_read(
        'https://drive.google.com/file/d/1YMWPJUlFDCm-EkjivaafcZNUUNyeIfsR/view?usp=drive_link',
        usecols=cols
    )
    # Display the DataFrame
    print(geo_df.head())
    
except ValueError as e:
    print(f"An error occurred: {e}")

    #===================================================================================================

# Initialize Faker
fake = Faker()

# Get a list of all countries
countries = [country.name for country in pycountry.countries]

# Define top 50 travel destinations
top_travel_destinations = [
    "France", "Spain", "Italy", "Germany", "United Kingdom",
    "USA", "Australia", "Canada", "Japan", "Thailand",
    "Greece", "Portugal", "Netherlands", "Switzerland",
    "New Zealand", "Austria", "Belgium", "Sweden", "Ireland",
    "Singapore", "Malaysia", "Mexico", "Dubai", "South Africa",
    "Czech Republic", "Norway", "Finland", "Denmark", "Poland",
    "Brazil", "Argentina", "Chile", "India", "Indonesia",
    "Vietnam", "Philippines", "Turkey", "Russia", "Egypt",
    "Costa Rica", "Peru", "Colombia", "Hong Kong", "Israel",
    "Morocco", "Croatia", "Hungary", "Slovakia", "Bulgaria",
]

# Define other necessary fields
travel_cover_types = ["Single Trip", "Annual Multi-Trip", "Backpacker"]
traveller_cover_types = ["Individual", "Couple", "Family", "Group"]
extra_cover_options = ["Gadgets", "Winter Sports", "Business Trip", "Cruise", "Adventure", "Digital Nomad", "Identity Theft"]
titles = ["Mr", "Mrs", "Miss", "Ms", "Dr", "Mx"]

def generate_customer_id():
    """Generate a unique customer ID."""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()  # Hash of a UUID

def generate_credit_scores(occupations, occupation_scores):
    """Generate credit scores based on occupations."""
    return [np.random.randint(occupation_scores[occupation][0], occupation_scores[occupation][1] + 1) for occupation in occupations]

#def calculate_age(dob):
#    """Calculate age from date of birth."""
#    today = pd.to_datetime('today')
#    return (today - dob).days // 365

# Generate synthetic travel insurance quote data
def generate_travel_quote_data(geo_df, n_rows=10000):
    postcode_sample = geo_df.sample(n=n_rows, replace=True, random_state=42).reset_index(drop=True)

    occupations = ['Professional', 'Manager', 'Skilled Worker', 'Unskilled Worker', 'Self-Employed']
    occupation_scores = {
        'Professional': (800, 999),
        'Manager': (700, 799),
        'Skilled Worker': (600, 699),
        'Unskilled Worker': (500, 599),
        'Self-Employed': (600, 750)
    }

    # Define coverage distribution probabilities
    coverage_distribution = ["silver", "gold", "platinum"]
    coverage_probs = [0.7, 0.2, 0.1]  # 70% silver, 20% gold, 10% platinum
    
    coverage = np.random.choice(
        coverage_distribution,
        size=n_rows,
        p=coverage_probs
    )
    
    random_occupations = np.random.choice(occupations, size=n_rows)
    credit_scores = generate_credit_scores(random_occupations, occupation_scores)
    
    # Define the current date
    current_date = datetime.now()

    # Define the age limits
    min_age = 16
    max_age = 77
    
    # Current date
    current_date = datetime.now()
    
    # DOB bounds
    # Convert to numeric (timestamps in ns)
    min_ts = pd.Timestamp(min_dob).value // 10**9
    max_ts = pd.Timestamp(max_dob).value // 10**9

    # Generate random DOBs
    random_timestamps = np.random.randint(min_ts, max_ts, n_rows)
    traveller_1_dob = pd.to_datetime(random_timestamps, unit='s')
    
    # Calculate age function
    def calculate_age(dobs):
        today = datetime.today()
        return [(today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))) for dob in dobs]
    
    ages = calculate_age(traveller_1_dob)

    df = pd.DataFrame({
        "quote_id": [str(uuid.uuid4()) for _ in range(n_rows)],
        "customer_id": [generate_customer_id() for _ in range(n_rows)],  # Generate customer IDs
        "quote_date": pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D'),
        "title": np.random.choice(titles, size=n_rows),
        "first_name": [fake.first_name() for _ in range(n_rows)],
        "last_name": [fake.last_name() for _ in range(n_rows)],
        "email_address": [fake.email() for _ in range(n_rows)],
        "traveller_1_dob": traveller_1_dob,
        "coverage": coverage,
        "age": ages,
        "occupation": random_occupations,
        "credit_score": credit_scores,
        "start_date": pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D'),
        "return_date": pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(1, 30, n_rows), unit='D'),
        "travel_cover_type": np.random.choice(travel_cover_types, size=n_rows),
        "destination": np.random.choice(top_travel_destinations, size=n_rows),
        "traveller_cover": np.random.choice(traveller_cover_types, size=n_rows),
        "extra_cover": np.random.choice(extra_cover_options, size=n_rows),
        "medical_declaration": np.random.choice(['Yes', 'No'], size=n_rows),
        "promotional_code": np.random.choice([0, 1], size=n_rows),
        "membership": np.random.choice(['None', 'Silver', 'Gold', 'Platinum'], size=n_rows),
        "postcode": postcode_sample["pcd"].values,
        "country": postcode_sample["country"].values,
        "region": postcode_sample["region"].values,
        "admin_district": postcode_sample["admin_district"].values,
    })

    print(f"✅ Generated {n_rows} synthetic travel insurance quote records with {df.shape[1]} columns")
    return df

# Load geo_df (ensure this is the correct path for your postcode data)
#cols = ['pcd', 'country', 'region', 'admin_district']
#geo_df = pd.read_csv('physar/data/geo/postcodes_api.csv', usecols=cols)
#geo_df = geo_df.dropna(subset=['country'])

#==========================================================

class PolicyDataCreator:
    def __init__(self, quote_df: pd.DataFrame, seed: int = 42):
        self.quote_df = quote_df
        self.seed = seed
        np.random.seed(self.seed)

    def generate_policy_data(self, customer_ids: list):
        # Filter the quote DataFrame for the specified customer IDs
        policy_candidates = self.quote_df[self.quote_df['customer_id'].isin(customer_ids)]

        # Generate the Policy_IDs without a loop
        policy_ids = [str(uuid.uuid4()) for _ in range(len(policy_candidates))]

        # Generate the policy DataFrame
        policy_data = pd.DataFrame({
            "Customer_ID": policy_candidates["customer_id"].values,
            "Policy_ID": policy_ids,  # Use pre-generated UUIDs
            "Age": policy_candidates["age"].values,
            "Occupation": policy_candidates["occupation"].values,
            "Credit_Score": policy_candidates["credit_score"].values,
            "Policy_Start_Date": self.random_policy_start_dates(policy_candidates["quote_date"]),
            "Policy_End_Date": pd.to_datetime(self.random_policy_start_dates(policy_candidates["quote_date"])) + pd.DateOffset(years=1),
            "Coverage": policy_candidates["coverage"].values,
            "Premium_Paid": self.calculate_premium(policy_candidates),
            #"Policy_Type": policy_candidates["policy_type"].values,
            #"New_Business_Flag": policy_candidates["policy_type"].apply(lambda x: 1 if x == "New" else 0)  # New Business Flag
        })

        return policy_data

    def random_policy_start_dates(self, quote_dates):
        return quote_dates + pd.to_timedelta(np.random.randint(0, 365, len(quote_dates)), unit='D')

    def calculate_premium(self, policy_candidates):
        # Calculate the premium based on age, occupation, and credit score
        base_premiums = np.full(len(policy_candidates), 300)  # Base premium

        # Adjustments based on age
        age_adjustments = np.where(policy_candidates['age'].values < 25, 200, 0) + \
                          np.where((policy_candidates['age'].values >= 25) & (policy_candidates['age'].values < 40), 100, 0)
        base_premiums += age_adjustments

        # Adjustments based on credit score
        score_adjustments = np.where(policy_candidates['credit_score'].values < 600, 100, 0) - \
                            np.where(policy_candidates['credit_score'].values >= 800, 50, 0)
        base_premiums += score_adjustments

        premiums = np.clip(np.random.normal(base_premiums, 50), 100, None)  # Ensure a minimum premium
        return premiums

#=============================================================================

class ClaimsDataCreator:
    def __init__(self, policy_df: pd.DataFrame, seed: int = 42):
        self.policy_df = policy_df
        self.seed = seed
        np.random.seed(self.seed)

    def generate_claims(self, n_claims: int):
        """Generate a specified number of claims across customer policies."""
        claims_data = []

        # Select unique customer IDs from policy_df
        customer_ids = self.policy_df['Customer_ID'].unique()

        for _ in range(n_claims):
            # Randomly select a customer ID
            customer_id = np.random.choice(customer_ids)

            # Select policies for the customer
            customer_policies = self.policy_df[self.policy_df['Customer_ID'] == customer_id]

            # Randomly select a policy for the claim
            if not customer_policies.empty:
                policy = customer_policies.sample(n=1).iloc[0]

                # Determine the number of claims (1, 2, or 3) based on distribution
                num_claims = self.determine_claims_count()

                for _ in range(num_claims):
                    # Generate claim details
                    claim_date = self.generate_claim_date(policy['Policy_Start_Date'], policy['Policy_End_Date'])
                    claim_amount = np.random.randint(100, 5000)  # Random claim amount
                    claim_cost = claim_amount * np.random.uniform(0.5, 1.5)  # Actual cost can vary
                    exposure = (policy['Policy_End_Date'] - claim_date).days

                    claims_data.append({
                        "claim_id": str(uuid.uuid4()),
                        "Customer_ID": customer_id,
                        "Policy_ID": policy['Policy_ID'],
                        "claim_date": claim_date,
                        "claim_amount": claim_amount,
                        "claim_cost": claim_cost,
                        "claim_description": "Claim for accident or damage",
                        "claim_status": np.random.choice(['Pending', 'Approved', 'Rejected']),
                        "exposure": exposure,
                        "claim_type": np.random.choice(['Accident', 'Theft', 'Weather']),
                        "claim_resolution_date": claim_date + timedelta(days=np.random.randint(1, 30)),
                        "claim_notes": "Additional notes regarding the claim."
                    })

        claims_df = pd.DataFrame(claims_data)
        return claims_df

    def determine_claims_count(self):
        """Determine the number of claims (1, 2, or 3) based on distribution."""
        rand_value = np.random.rand()
        if rand_value < 0.6:
            return 1  # 60% of customers will have 1 claim
        elif rand_value < 0.9:
            return 2  # 30% of customers will have 2 claims
        else:
            return 3  # 10% of customers will have 3 claims

    def generate_claim_date(self, start_date, end_date):
        """Generate a claim date within the policy period."""
        delta_days = (end_date - start_date).days
        claim_date = start_date + timedelta(days=np.random.randint(1, delta_days))
        return claim_date

#=========================================================

# Generate data
travel_quotes = generate_travel_quote_data(geo_df, n_rows=10000)

# Optionally save to a CSV file
#travel_quotes.to_csv("synthetic_travel_quotes.csv", index=False)
#print("📁 Saved dataset to synthetic_travel_quotes.csv")

# Determine 5% of the DataFrame
sample_size = int(len(travel_quotes) * 0.05)  # Calculate 5% of the DataFrame
sampled_quote_df = travel_quotes.sample(n=sample_size, random_state=42)  # Sample 5%

# Get unique customer IDs from the sampled DataFrame
customer_ids = sampled_quote_df["customer_id"].unique()

# Create policy data
policy_creator = PolicyDataCreator(sampled_quote_df)
%time policy_df = policy_creator.generate_policy_data(customer_ids)


print(travel_quotes.head())
print(policy_df.head())

# Create the ClaimsDataCreator instance
claims_creator = ClaimsDataCreator(policy_df)

# Generate a specified number of claims
n_claims = 7000  # Specify the number of claims you want to generate
claims_df = claims_creator.generate_claims(n_claims)

claims_df.head(5)
