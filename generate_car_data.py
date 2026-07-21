"""
Generate realistic car price dataset for price prediction task.
"""

import numpy as np
import pandas as pd


def generate_car_data(n_samples=1000, output_file="car_data.csv"):
    """
    Generate a realistic car dataset with features and prices.
    
    Parameters:
    - n_samples: Number of car records to generate
    - output_file: Path to save the CSV file
    """
    np.random.seed(42)
    
    # Brand configurations: (base_price_multiplier, reliability_score, popularity_score)
    brands = {
        'Toyota': (1.0, 0.85, 0.90),
        'Honda': (0.95, 0.83, 0.85),
        'Ford': (0.85, 0.75, 0.80),
        'Chevrolet': (0.80, 0.72, 0.78),
        'BMW': (1.8, 0.78, 0.75),
        'Mercedes': (2.0, 0.80, 0.77),
        'Audi': (1.7, 0.76, 0.73),
        'Tesla': (3.5, 0.82, 0.88),
        'Hyundai': (0.75, 0.80, 0.76),
        'Kia': (0.70, 0.78, 0.74),
        'Nissan': (0.78, 0.74, 0.72),
        'Volkswagen': (0.88, 0.77, 0.75),
        'Subaru': (1.1, 0.84, 0.70),
        'Mazda': (0.92, 0.79, 0.68),
    }
    
    data = []
    
    for _ in range(n_samples):
        brand = np.random.choice(list(brands.keys()))
        base_mult, reliability, popularity = brands[brand]
        
        # Generate car features
        year = np.random.randint(2010, 2024)
        age = 2024 - year
        
        # Engine specifications
        engine_size = np.random.choice([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        horsepower = int(engine_size * 60 + np.random.normal(0, 15))
        horsepower = max(60, min(600, horsepower))
        
        # Mileage (higher for older cars)
        mileage = age * 12000 + np.random.normal(0, 10000)
        mileage = max(0, int(mileage))
        
        # Fuel efficiency (higher for newer/smaller engines)
        fuel_efficiency = 50 - engine_size * 5 - age * 0.5 + np.random.normal(0, 3)
        fuel_efficiency = max(10, min(60, fuel_efficiency))
        
        # Number of previous owners
        if age <= 2:
            owners = 1
        elif age <= 5:
            owners = np.random.randint(1, 3)
        elif age <= 8:
            owners = np.random.randint(1, 4)
        else:
            owners = np.random.randint(2, 6)
        
        # Condition rating (1-10)
        condition = max(1, min(10, 10 - age * 0.5 + reliability * 3 + np.random.normal(0, 1)))
        condition = round(condition, 1)
        
        # Fuel type
        fuel_type = np.random.choice(['Petrol', 'Diesel', 'Hybrid', 'Electric'], 
                                     p=[0.5, 0.3, 0.15, 0.05])
        
        # Transmission
        transmission = np.random.choice(['Manual', 'Automatic'], p=[0.3, 0.7])
        
        # Calculate price
        base_price = 20000
        price = base_price * base_mult
        
        # Age depreciation
        price *= np.exp(-0.12 * age)
        
        # Mileage depreciation
        price *= max(0.6, 1 - (mileage / 200000))
        
        # Engine size adjustment
        price *= (1 + (engine_size - 2.0) * 0.1)
        
        # Horsepower adjustment
        price *= (1 + (horsepower - 150) * 0.001)
        
        # Condition adjustment
        price *= (condition / 7)
        
        # Transmission adjustment
        if transmission == 'Automatic':
            price *= 1.05
        
        # Fuel type adjustment
        if fuel_type == 'Electric':
            price *= 1.3
        elif fuel_type == 'Hybrid':
            price *= 1.15
        elif fuel_type == 'Diesel':
            price *= 1.02
        
        # Brand-specific additional adjustments
        if brand in ['BMW', 'Mercedes', 'Audi']:
            price *= (1 + age * 0.02)  # Luxury brands depreciate slower
        elif brand in ['Toyota', 'Honda']:
            price *= 1.05  # Reliability premium
        
        # Add random noise
        price *= np.random.normal(1, 0.05)
        
        # Round price to nearest 100
        price = round(price / 100) * 100
        price = max(1000, price)
        
        data.append({
            'Brand': brand,
            'Year': year,
            'Age': age,
            'Engine_Size': engine_size,
            'Horsepower': horsepower,
            'Mileage': mileage,
            'Fuel_Efficiency': round(fuel_efficiency, 1),
            'Owners': owners,
            'Condition': condition,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Price': price
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Car data saved to '{output_file}'")
    print(f"  Records: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Brands: {df['Brand'].nunique()}")
    print(f"  Price range: ${df['Price'].min():,} - ${df['Price'].max():,}")
    print(f"  Mean price: ${df['Price'].mean():,.0f}")
    
    return df


if __name__ == "__main__":
    generate_car_data()

