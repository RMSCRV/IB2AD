# Raw Data Directory

## Dataset Information

This project uses the **California Housing Dataset** which is built into scikit-learn and is automatically loaded when running the notebook.

### About the Dataset

- **Source**: 1990 California census
- **Samples**: 20,640 housing districts
- **Features**: 8 numerical features
- **Target**: Median house value (in $100,000s)

### Features

1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

### Target Variable

- **MedHouseVal**: Median house value for California districts (in $100,000s)

## Using Your Own Dataset

To use your own dataset instead:

1. Place your CSV file in this directory (e.g., `my_data.csv`)
2. Open `ml_pipeline.ipynb`
3. In the "Data Loading" section, replace:
   ```python
   housing_data = fetch_california_housing(as_frame=True)
   df = housing_data.frame
   ```

   With:
   ```python
   df = pd.read_csv('./data/raw/my_data.csv')
   ```

4. Update the target column name if different from 'MedHouseVal'
5. Adjust feature engineering based on your domain

## Data Files

The notebook will automatically:
- Load the California Housing dataset from scikit-learn
- Save cleaned data to `../processed/cleaned_housing_data.csv`
- Generate sample predictions and results

No manual data download required!
