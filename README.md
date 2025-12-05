# IB2AD0_Data_Science_GenerativeAI
Notebooks for the IB2AD0 Data Science &amp; Generative AI module 2025/26

---

## Machine Learning Pipeline Project

Comprehensive end-to-end machine learning pipelines demonstrating data preprocessing, feature engineering, model training, and evaluation.

### Project Structure

```
├── ml_exam.ipynb          # Comprehensive exam notebook (RECOMMENDED)
├── ml_pipeline.ipynb      # Alternative comprehensive notebook
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

#### For Google Colab (Recommended for ml_exam.ipynb):
1. Upload `ml_exam.ipynb` to Google Colab
2. Click "Open in Colab" badge in the notebook
3. Run cells sequentially from top to bottom

#### For Local Jupyter:
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook ml_exam.ipynb
   # or
   jupyter notebook ml_pipeline.ipynb
   ```

3. **Run All Cells**
   - Execute cells sequentially from top to bottom
   - All outputs and visualizations will be generated inline

### Features

#### ml_exam.ipynb (Exam-Ready Notebook)
- **7 Different Models**: Linear Regression, Random Forest Regressor, Decision Tree, Random Forest Classifier, GBDT, PyTorch Logistic Regression
- **Binning Strategy**: Converts continuous target to classification problem
- **Macro Metrics**: Precision, Recall, F1-score for all classifiers
- **Multiclass Confusion Matrices**: Detailed confusion matrices for all classification models
- **Following Course Style**: References 5_01 (GBDT), 6_01 (PyTorch LogReg), 5_02 (Hackathon)
- **Google Colab Ready**: Designed for step-by-step execution

#### ml_pipeline.ipynb (Alternative Notebook)
- **3 Models**: Random Forest, Ridge Regression, PyTorch Neural Network
- **Regression Focus**: Continuous target prediction
- **Standard Metrics**: MAE, MSE, RMSE, R²

#### Both Notebooks Include:
- **Comprehensive Data Exploration**: Statistical analysis, missing value detection, distribution analysis
- **Data Cleaning**: Handling missing values, outliers, duplicates
- **Feature Engineering**: Creating new features, encoding, scaling (14 features from 8)
- **Model Evaluation**: Detailed metrics, visualizations, and comparisons
- **Reproducibility**: All random seeds set, complete documentation

### Models Implemented

#### ml_exam.ipynb:
1. **Linear Regression**: Standard regression baseline
2. **Random Forest Regressor**: Ensemble regression method
3. **Decision Tree Classifier**: Tree-based classification
4. **Random Forest Classifier**: Ensemble classification method
5. **GBDT**: Gradient Boosting Decision Trees (following 5_01)
6. **PyTorch Logistic Regression**: Neural network classification (following 6_01)

#### ml_pipeline.ipynb:
1. **Random Forest**: Ensemble method for regression
2. **Ridge Regression**: Regularized linear regression
3. **PyTorch Neural Network**: Deep learning model

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
