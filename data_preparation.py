import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting data preparation...")

def prepare_price_prediction_data():
    """Prepare and clean the price prediction dataset"""
    
    # Load the data
    df = pd.read_csv('maharashtra_districts_complete.csv')
    print(f"\n📊 Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # === DATA CLEANING ===
    
    # 1. Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"\n✅ Removed {initial_rows - len(df)} duplicate rows")
    
    # 2. Handle missing values
    print("\n📉 Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Drop rows with missing critical values
    critical_cols = ['Commodity', 'District', 'Market', 'Variety', 'Grade', 
                     'Min_Price', 'Max_Price', 'Modal_Price', 'Arrival_Date']
    
    df = df.dropna(subset=critical_cols)
    print(f"\n✅ Dropped rows with missing critical values. Remaining: {len(df)}")
    
    # 3. Fix data types and convert dates
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Arrival_Date'])
    
    # 4. Ensure price columns are numeric and positive
    for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] > 0]  # Remove zero or negative prices
        
        # Remove extreme outliers (prices beyond 3 standard deviations)
        mean_val = df[col].mean()
        std_val = df[col].std()
        df = df[df[col].between(mean_val - 3*std_val, mean_val + 3*std_val)]
    
    # 5. Ensure Min <= Modal <= Max
    df = df[df['Min_Price'] <= df['Modal_Price']]
    df = df[df['Modal_Price'] <= df['Max_Price']]
    
    print(f"\n📊 After cleaning: {len(df)} rows")
    
    # === FEATURE ENGINEERING ===
    
    # Extract date features
    df['Year'] = df['Arrival_Date'].dt.year
    df['Month'] = df['Arrival_Date'].dt.month
    df['Day'] = df['Arrival_Date'].dt.day
    df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
    df['Quarter'] = df['Arrival_Date'].dt.quarter
    
    # Add season feature (Indian agricultural seasons)
    def get_season(month):
        if month in [6, 7, 8, 9]:  # Kharif (monsoon crops)
            return 'Kharif'
        elif month in [10, 11, 12, 1]:  # Rabi (winter crops)
            return 'Rabi'
        elif month in [2, 3, 4, 5]:  # Summer
            return 'Summer'
        else:
            return 'Other'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Price ratios (useful features)
    df['Price_Range'] = df['Max_Price'] - df['Min_Price']
    df['Price_Volatility'] = df['Price_Range'] / df['Modal_Price']
    
    # In data_preparation.py, fix the lag features section:

# === FIXED: Create lag features properly (only use PAST data) ===
    df_sorted = df.sort_values(['District', 'Commodity', 'Arrival_Date'])

    # Create lag features using shift (this uses PREVIOUS rows, not future)
    df_sorted['Prev_Modal_Price'] = df_sorted.groupby(['District', 'Commodity'])['Modal_Price'].shift(1)
    df_sorted['Prev_2_Modal_Price'] = df_sorted.groupby(['District', 'Commodity'])['Modal_Price'].shift(2)
    df_sorted['Prev_3_Modal_Price'] = df_sorted.groupby(['District', 'Commodity'])['Modal_Price'].shift(3)

    # Calculate changes using only past data
    df_sorted['Price_Change'] = df_sorted['Modal_Price'] - df_sorted['Prev_Modal_Price']
    df_sorted['Price_Change_Pct'] = (df_sorted['Price_Change'] / df_sorted['Prev_Modal_Price']) * 100

    # Moving average using ONLY past data (not centered)
    df_sorted['Modal_Price_MA3'] = df_sorted.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()  # Use shift to ensure only past data
    )

    # Price momentum using past data only
    df_sorted['Price_Momentum'] = df_sorted['Modal_Price'] / df_sorted['Modal_Price_MA3']

    # Drop rows with NaN in lag features (first few rows per group)
    df_sorted = df_sorted.dropna(subset=['Prev_Modal_Price', 'Modal_Price_MA3'])

    df = df_sorted.copy()
    
    # Rolling averages (3-month)
    df['Modal_Price_MA3'] = df_sorted.groupby(['District', 'Commodity'])['Modal_Price'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Price momentum
    df['Price_Momentum'] = df['Modal_Price'] / df['Modal_Price_MA3']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['District', 'Market', 'Commodity', 'Variety', 'Grade', 'Season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n🔄 Final dataset shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    
    return df, label_encoders

def prepare_crop_recommendation_data():
    """Create synthetic crop recommendation data based on price data"""
    
    # Load price data to understand crop distribution
    price_df = pd.read_csv('maharashtra_districts_complete.csv')
    
    # Get unique crops and their characteristics
    crops = price_df['Commodity'].unique()
    
    # Create synthetic crop recommendation dataset
    np.random.seed(42)
    n_samples = 5000
    
    recommendation_data = []
    
    for i in range(n_samples):
        # Generate random environmental conditions
        temperature = np.random.uniform(10, 40)
        humidity = np.random.uniform(30, 90)
        rainfall = np.random.uniform(50, 300)
        nitrogen = np.random.uniform(0, 140)
        phosphorus = np.random.uniform(0, 145)
        potassium = np.random.uniform(0, 205)
        ph = np.random.uniform(4.5, 8.5)
        
        # Select a random crop
        crop = np.random.choice(crops)
        
        # Adjust conditions based on crop type
        if 'Rice' in crop:
            if np.random.random() < 0.8:  # 80% chance to be suitable
                temperature = np.random.uniform(20, 35)
                rainfall = np.random.uniform(100, 250)
                humidity = np.random.uniform(60, 85)
        elif 'Wheat' in crop:
            if np.random.random() < 0.8:
                temperature = np.random.uniform(15, 25)
                rainfall = np.random.uniform(50, 150)
        elif 'Cotton' in crop:
            if np.random.random() < 0.8:
                temperature = np.random.uniform(25, 35)
                rainfall = np.random.uniform(50, 100)
        
        recommendation_data.append({
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 1),
            'nitrogen': round(nitrogen, 1),
            'phosphorus': round(phosphorus, 1),
            'potassium': round(potassium, 1),
            'ph': round(ph, 1),
            'label': crop
        })
    
    rec_df = pd.DataFrame(recommendation_data)
    
    # Shuffle
    rec_df = rec_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n🌾 Crop recommendation dataset created: {rec_df.shape}")
    print(f"Unique crops: {rec_df['label'].nunique()}")
    
    return rec_df

if __name__ == "__main__":
    # Prepare price prediction data
    price_df, encoders = prepare_price_prediction_data()
    price_df.to_csv('price_prediction_cleaned.csv', index=False)
    print(f"\n✅ Price data saved to 'price_prediction_cleaned.csv'")
    
    # Prepare crop recommendation data
    rec_df = prepare_crop_recommendation_data()
    rec_df.to_csv('crop_recommendation_data.csv', index=False)
    print(f"✅ Recommendation data saved to 'crop_recommendation_data.csv'")