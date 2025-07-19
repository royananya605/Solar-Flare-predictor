# Solar-Flare-predictor

This project uses a deep neural network built with TensorFlow/Keras to predict the intensity of different solar flare types (C, M, X, S) based on solar activity data. It includes data preprocessing, normalization, model training, prediction, and visualization of error distributions.

---

## 📊 Dataset

The dataset `solar_flare.csv` should contain the following columns:

| Column Name               | Description                                 |
|--------------------------|---------------------------------------------|
| date in number format    | Date converted to numerical format          |
| mean_mag_field           | Average magnetic field strength             |
| total_solar_irradiance   | Total solar irradiance                      |
| radio_flux_density       | Solar radio flux                            |
| sunspot_number           | Number of sunspots                          |
| sunspot_area_10e-6hemis  | Area of sunspots                            |
| xray_flux                | X-ray flux measurement                      |
| lyman_alpha              | Lyman-alpha emission                        |
| flare_c, flare_m, flare_x, flare_s | C, M, X, S class flare intensities |
| Flare_total              | Total number of flares                      |

---

##  How It Works

###  Preprocessing
- Selected input features and flare targets are extracted.
- Min-max normalization is applied to scale all values between 0 and 1.

###  Model Architecture

A dense neural network model with the following structure:

- Input: 12 features  
- Hidden Layers:
  - Dense(256, ReLU)
  - Dense(1024, ReLU)
  - Dense(1024, ReLU)
  - Dense(128, ReLU)
- Output: Dense(8, ReLU) *(though only 4 values are used in evaluation)*

Optimizer: SGD with Nesterov momentum  
Loss: Mean Squared Logarithmic Error (MSLE)  
Callback: Early stopping to avoid overfitting  

### 📈 Training
- Training set: First 5476 samples
- Test set: From index 5477 onward
- Validation split: 20% of training set
- Epochs: Up to 3000
- Batch size: 100

---

##  Installation

Install required dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
