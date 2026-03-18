Employee Attendance Anomaly Detection

Project Overview
This project analyzes employee attendance behavior using Machine Learning to detect abnormal working patterns. The system processes employee login and logout records, performs feature engineering, and identifies anomalies using anomaly detection models.
The application is built as an interactive web dashboard using Streamlit, allowing users to upload attendance datasets and visualize insights.

Problem Statement
Organizations collect employee attendance data but often lack intelligent tools to analyze behavioral patterns.

This system helps detect:
- Employees leaving work significantly earlier than expected
- Unusually short working hours
- Abnormal attendance patterns compared to normal behavior

The goal is to provide data-driven insights into employee attendance behavior.

Technologies Used
- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn

Machine Learning Approach

The system uses two anomaly detection algorithms.

Isolation Forest  
Detects global anomalies by isolating unusual data points in the dataset.

Local Outlier Factor (LOF)  
Detects local density deviations and identifies records that behave differently from their neighboring data points.

The final anomaly decision is made using the combined output of both models.

Feature Engineering

Raw attendance data is transformed into meaningful numerical features.

The following features are created:

- login_minutes
- logout_minutes
- total_work_minutes
- work_hours
- day_of_week
- department
- login_deviation from standard office time

These features allow the machine learning model to understand employee attendance behavior.

Streamlit Application Features

Dataset Upload  
Users can upload a CSV attendance dataset.

Data Preview  
Displays the uploaded dataset for quick inspection.

Feature Engineering Output  
Shows processed features used for machine learning.

Model Training  
Runs anomaly detection models on the dataset.

Results Visualization  
The dashboard provides several visual insights including:

- Work duration distribution
- Login time vs work duration
- Number of anomalies detected
- Anomaly counts by employee
- Anomaly counts by department
- Attendance trend analysis

Anomaly Detection Output  
The system highlights abnormal attendance records along with possible reasons.

Example Output

Example anomaly detection result:

Employee ID: E017  
Login Time: 09:05  
Logout Time: 15:40  
Work Duration: 6.5 hours  
Status: High Anomaly  
Reason: Short Work Duration

Running the Application Locally

Step 1: Clone the repository

git clone https://github.com/Aparna-26-02/ai-attendance-anomaly-detection.git

Step 2: Install dependencies

pip install -r requirements.txt

Step 3: Run the Streamlit application

streamlit run streamlit_app.py

Live Demo

Streamlit App Link:

https://ai-attendance-anomaly-detection-crekcwtdnffrg2vwdmkkjn.streamlit.app/

Repository Structure

ai-attendance-anomaly-detection/

streamlit_app.py  
requirements.txt  
README.md  
.gitignore
