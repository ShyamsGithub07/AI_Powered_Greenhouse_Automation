# 🌱 AI-Powered Greenhouse Automation

## Overview
An **AI-powered control system** has been developed for ecosystems in greenhouses.  
It utilizes the **two neural network models**:
- **Classification model** — determines the actions (FAN, IRRIGATE, SHADE, NONE).  
- **Regression model** — predicts exact actuator values (e.g. fan speed, irrigation amount).  

The system is capable of being utilized for **independent decision-making** where the parameters like temperature, humidity, light, soil moisture, and CO₂ levels are adjusted accordingly to create the best growing conditions for the crops.

---

## 🧠 Project Objectives
- Greenhouse management powered by artificial intelligence based on the data.  
- Co-control action prediction and actuator output refinement.  
- TensorFlow SavedModel format for easy deployment.  

---

## ⚙️ Tech Stack
- **Languages:** Python  
- **Libraries:** TensorFlow / Keras, scikit-learn, NumPy, Pandas, Matplotlib  
- **Modeling Techniques:** Neural Networks, StandardScaler normalization, EarlyStopping regularization  
- **Deployment:** TensorFlow SavedModel format (cloud or local integration)  

---

## 📊 Dataset
**Features:**
- Temperature (`temp_c`)  
- Relative Humidity (`rh`)  
- Light Intensity (`light_lux`)  
- Soil Moisture (`soil_moisture`)  
- CO₂ Concentration (`co2`)

**Targets:**
- `target_action` — classification label (FAN, IRRIGATE, SHADE, NONE)  
- `fan_speed`, `irrigation_ml` — regression outputs (continuous control values)

The script will automatically **create synthetic data** for testing if no CSV file is uploaded.

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib joblib
   ```

2. **Run the project**
   ```bash
   python greenhouse_ai_.py
   ```

3. **Outputs:**
   - Trained models saved under the `models/` folder:
     - `greenhouse_classifier/`
     - `greenhouse_regressor/`
   - Scalers and encoders:
     - `clf_scaler.joblib`, `label_encoder.joblib`, etc.  

4. **Inference Example**
   ```python
   sample_obs = {
       'temp_c': 30.0,
       'rh': 45.0,
       'light_lux': 30000,
       'soil_moisture': 25.0,
       'co2': 420.0
   }
   print(infer_classification(sample_obs))
   print(infer_regression(sample_obs))
   ```

---

## 🧩 Model Training Details
- **Preprocessing:** StandardScaler normalization  
- **Classification:** Softmax neural network with dropout regularization  
- **Regression:** Multi-output MLP with linear activation for continuous predictions  
- **Callbacks:** EarlyStopping (patience=10) for overfitting prevention  
- **Metrics:** Accuracy (classification), MAE / RMSE (regression)  

---

## 📈 Example Results
| Model | Metric | Score |
|--------|---------|--------|
| Classification | Accuracy | ~92% |
| Regression | MAE | 0.05 |
| Regression | RMSE | 0.08 |

(*Replace with your actual values after training.*)

---

##  Future Enhancements
- Real IoT sensors such as DHT11, CO₂, LDR, soil probe will be integrated to perform live inference.  
- Raspberry Pi or ESP32 with TensorFlow Lite will be the target deployment platform.  
- Adaptive greenhouse control will include reinforcement learning as one of its main components.


---

## 📂 Project Structure
```
├── greenhouse_ai_.py          # Main training and inference script
├── greenhouse_data_.csv       
├── README.md
```
