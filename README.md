# Using Regression Models for Atmospheric Retrieval of Exoplanets

## Project Overview
This project demonstrates the application of machine learning (ML) techniques to classify and characterize the atmospheres of exoplanets based on their spectral features. We analyzed a dataset of synthetic exoplanetary spectra to retrieve atmospheric parameters using advanced ML models, including Extra Tree Regressor, Random Forest Regressor, and XGBoost Regressor.

## Key Features
- **Dataset**: Synthetic spectra generated with the Heng & Kitzmann (2017) model, binned to 13 wavelength bins corresponding to Hubble Space Telescope WFC3.
- **ML Models**: Ensemble-based models like Extra Tree and Random Forest for high accuracy and efficiency.
- **Dimensionality Reduction**: Incremental PCA (IPCA) for feature compression.
- **Application**: Retrieval of atmospheric parameters for the exoplanet WASP-12b.

---

## Dataset
The dataset consists of:
- **Spectral Range**: 0.8 - 1.7 µm.
- **Features**: 13 spectral features and 5 atmospheric parameters: temperature (T), volume mixing ratios of water (XH2O), ammonia (XNH3), hydrogen cyanide (XHCN), and cloud opacity (κ0).

### Data Distribution
| Parameter   | Lower Bound | Upper Bound | Prior         |
|-------------|-------------|-------------|---------------|
| T (K)       | 500         | 2900        | Uniform       |
| log XH2O    | 10^-13      | 1           | Log-uniform   |
| log XHCN    | 10^-13      | 1           | Log-uniform   |
| log XNH3    | 10^-13      | 1           | Log-uniform   |
| log κ0      | 10^-13      | 10^2        | Log-uniform   |

---

## Methodology

1. **Model Selection**:
   - **PyCaret** was used to compare regression models.
   - Top models: Extra Tree Regressor, Random Forest Regressor, and XGBoost Regressor.

2. **Preprocessing**:
   - Data normalization using `StandardScaler` and `MinMaxScaler` from Scikit-learn.
   - Incremental PCA for dimensionality reduction.

3. **Training and Evaluation**:
   - Split dataset into 80,000 training and 20,000 testing samples.
   - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R²).

4. **Hyperparameter Tuning**:
   - Performed using GridSearchCV.
   - Focused on Extra Tree Regressor and XGBoost for improved performance.

---

## Results

### Model Performance (Without PCA)
| Model        | T (K) R² | log XH2O R² | log XHCN R² | log XNH3 R² | log κ0 R² |
|--------------|----------|-------------|-------------|-------------|-----------|
| RF (MN18)    | 0.746    | 0.608       | 0.467       | 0.700       | 0.737     |
| RFR          | 0.7471   | 0.6058      | 0.467       | 0.6997      | 0.737     |
| ETR          | 0.7524   | 0.6083      | 0.4708      | 0.705       | 0.7392    |
| XGBR         | 0.7344   | 0.5878      | 0.4541      | 0.6796      | 0.7322    |

### Model Performance (With PCA)
| Model        | T (K) R² | log XH2O R² | log XHCN R² | log XNH3 R² | log κ0 R² |
|--------------|----------|-------------|-------------|-------------|-----------|
| RF (MN18)    | 0.746    | 0.608       | 0.467       | 0.700       | 0.737     |
| RFR (IPCA)   | 0.7602   | 0.6200      | 0.4827      | 0.7149      | 0.7419    |
| ETR (IPCA)   | 0.7655   | 0.6205      | 0.4867      | 0.7203      | 0.7441    |

### Predictions for WASP-12b (ETR Model)
| Parameter    | Retrieved Value |
|--------------|-----------------|
| T (K)        | 1058.56         |
| log XH2O     | -2.56           |
| log XHCN     | -6.33           |
| log XNH3     | -8.20           |
| log κ0       | -2.29           |

---

## Visuals
#### Figure : True vs Predicted Temperatures
![Model Predictions](https://github.com/user-attachments/assets/a6f42a5d-f14c-4bfc-a977-92405dddd490)


---

## Improvements
1. **Dimensionality Reduction**: IPCA enhanced R² values for most parameters.
2. **Hyperparameter Tuning**: XGBoost showed marked improvements post-tuning.
3. **Model Blending**: Future work will explore model stacking for enhanced predictions.

---

## Future Directions
- Incorporate the Ariel ML Data Challenge 2023 dataset.
- Experiment with neural networks and clustering techniques.
- Use advanced preprocessing like anomaly detection for more robust parameter retrieval.

---

## References
1. Márquez-Neila, Pablo et al. *Supervised Machine Learning for Analyzing Spectra of Exoplanetary Atmospheres*. Nature Astronomy, 2018.
2. Nixon, M. C., Madhusudhan, N. *Assessment of Supervised Machine Learning for Atmospheric Retrieval of Exoplanets*. Monthly Notices of the Royal Astronomical Society, 2020.
3. Heng, K., Tsai, S.-M. *Analytical Models of Exoplanetary Atmospheres*. Astrophys. J., 2016.
