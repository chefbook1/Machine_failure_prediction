Objective :
To develop a predictive model that can accurately identify potential machine failures, enabling proactive maintenance and reducing downtime in industrial settings.

Dataset :
The dataset consists of 944 entries with 10 features:
footfall: Operational parameter
tempMode: Temperature mode setting
AQ: Air Quality measurement
USS: Ultrasonic Sensor reading
CS: Current Sensor reading
VOC: Volatile Organic Compounds measurement
RP: Resistance Parameter
IP: Input Parameter
Temperature: Temperature reading
fail: Target variable (1 = failure, 0 = no failure)

Technical Implementation :
Libraries Used :
pandas, numpy - Data manipulation
matplotlib, seaborn - Data visualization
scikit-learn - Machine learning models
imblearn - Handling class imbalance
joblib - Model serialization

Key Steps -
1.Data Loading & Exploration: Initial analysis of dataset structure and statistics
2.Data Visualization: Exploratory data analysis to understand feature distributions
3.Data Preprocessing:
Handling missing values
Addressing class imbalance using SMOTE
Feature scaling with StandardScaler
4.Model Development:
Random Forest Classifier
Hyperparameter tuning using GridSearchCV
5.Model Evaluation:
Classification reports
Confusion matrices
Accuracy scores
ROC-AUC metrics

How to Run -
Prerequisites:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

Execution :

    Clone the repository

    Ensure the dataset (data.csv) is in the project directory

    Run the Jupyter notebook: Machine_failure_project.ipynb

📈 Results

The project demonstrates a comprehensive approach to predictive maintenance, including:

    Data exploration and visualization

    Handling of imbalanced datasets

    Model training and optimization

    Performance evaluation metrics
