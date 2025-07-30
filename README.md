# Churn Prediction – Orange Telecom
## This project applies Machine Learning techniques to predict customer churn in a telecommunications company, based on historical data.

Using real telecom data, I developed a predictive system that identifies customers likely to **cancel their service**. This is a crucial tool for any company that wants to **reduce losses and boost retention**.


## Why It Matters

Keeping a customer is cheaper than getting a new one.  
With the right data, we can act before they leave.

This model helps companies:
- **Detect churn risk early**
- **Focus on high-risk customers**
- **Personalize offers and support**
- **Increase loyalty and revenue**


## Files in This Repo

- `Churn_Proyecto.ipynb` – Main notebook with the full analysis and modeling pipeline
- `modelo_RandomForest.ipynb` – detailed RF training
- `modelo_RedeNeural.ipynb` – neural network model (MLP)
- `modelo_RL.ipynb` - Logistics Regression model 


## Dataset
- `churn-bigml-80.csv` – main dataset for training
- `churn-bigml-20.csv` – held-out test set


## What I Did

1. **Exploratory Data Analysis (EDA)**  
   - Identified patterns and key churn indicators  
   - Handled missing data and outliers

2. **Balanced the dataset**  
   - The original data had a strong imbalance:  
     `2278` non-churn vs. only `388` churn cases  
   - Applied **undersampling** to create a fair training base

3. **Trained multiple models**
   - **Random Forest** (best performance)
   - MLPClassifier (Neural Network)
   - Logistics Regression 

4. **Tested on real, unseen data**
   - Final test used a separate dataset (`churn-bigml-20.csv`)  
   - Model generalized well and maintained performance



## Results (Random Forest)

| Metric            | Churn = 1 (Yes) |
|-------------------|-----------------|
| Precision         | 85%             |
| Recall (Sensitivity) | 87%          |
| Accuracy Overall  | 86%             |

> **High recall means we correctly identified most churners**  
> **High precision means we made few false churn predictions**



## Tech Stack

- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy` – data wrangling  
  - `matplotlib`, `seaborn` – visualization  
  - `scikit-learn` – machine learning models  
  - `MLPClassifier` (neural net), `RandomForestClassifier`  
- **Notebook:** Jupyter



### Future Improvements 💡 

1. Integrate the model into a live system to track active customers and generate real-time churn risk alerts
   
2. Explore more advanced models or AutoML solutions to further improve performance and scalability
   
3. Include additional data sources, like customer satisfaction surveys or support ticket sentiment, to enrich the model’s predictions


## Dataset Source

- This project uses the Orange Telecom Churn Dataset, available on Kaggle.
- The data includes customer profiles and churn information, already cleaned and split into training (churn-bigml-80.csv) and test (churn-bigml-20.csv) sets.

## Data Source

The data used in this project was obtained from Kaggle:
https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets

## Author

João Filipe
Data Science Enthusiast | Machine Learning Practitioner 
[GitHub Profile »](https://github.com/Joao-Filipe3)

---

⭐ If this project inspired you or helped, feel free to leave a star! :)
