# IB2AD0_Data_Science_GenerativeAI
Notebooks for the IB2AD0 Data Science &amp; Generative AI module 2025/26

---

## Machine Learning Pipeline Project

A comprehensive end-to-end machine learning pipeline demonstrating data preprocessing, feature engineering, model training, and evaluation.

### Project Structure

```
├── ml_pipeline.ipynb      # Main comprehensive notebook
├── examples/              # Reference repositories
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and processed data
├── models/               # Saved trained models
├── figures/              # Generated visualizations
├── src/
│   └── utils.py         # Helper functions
├── requirements.txt      # Python dependencies
└── README.md
```

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook ml_pipeline.ipynb
   ```

3. **Run All Cells**
   - Execute cells sequentially from top to bottom
   - All outputs and visualizations will be generated inline

### Features

- **Comprehensive Data Exploration**: Statistical analysis, missing value detection, distribution analysis
- **Data Cleaning**: Handling missing values, outliers, duplicates
- **Feature Engineering**: Creating new features, encoding, scaling
- **Multiple Models**: Random Forest, Logistic Regression, PyTorch Neural Network
- **Model Evaluation**: Detailed metrics, visualizations, and comparisons
- **Reproducibility**: All random seeds set, complete documentation

### Models Implemented

1. **Random Forest**: Ensemble method for classification/regression
2. **Logistic Regression**: Linear model for classification
3. **PyTorch Neural Network**: Deep learning model with customizable architecture

### Requirements

- Python 3.8+
- See `requirements.txt` for complete list of dependencies

### Usage

The `ml_pipeline.ipynb` notebook is self-contained and includes:
- All necessary imports and setup
- Sample dataset (or instructions to add your own)
- Complete pipeline from data loading to model evaluation
- Rich markdown documentation and explanations
- Inline visualizations and results

### Customization

To use your own dataset:
1. Place your data file in `data/raw/`
2. Update the data loading section in the notebook
3. Adjust feature engineering based on your domain
4. Run all cells

### Results

All results are saved to:
- `models/`: Trained model files
- `figures/`: Generated plots and visualizations
- `data/processed/`: Cleaned datasets

### License

See LICENSE file for details.
