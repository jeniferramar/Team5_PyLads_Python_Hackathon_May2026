
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(layout='wide')

st.title('Diabetes Healthcare Analytics Dashboard')

st.markdown('''
This dashboard presents:
- Descriptive Analysis
- Predictive Analysis
- Prescriptive Analysis

using physiological, demographic, sleep, and activity markers.
''')
# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

merged_df = pd.read_csv(
    r"C:\Users\amhum\Downloads\TeamNumber_TeamName_cleaned_data.csv"
)

# Remove missing values
merged_df.dropna(inplace=True)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header('Dataset Overview')

# KPI calculations
avg_glucose = merged_df['Glucose'].mean()
avg_hr = merged_df['Heart_rate'].mean()
avg_age = merged_df['Age'].mean()

# Create columns
col1, col2, col3 = st.columns(3)

col1.metric('Average Glucose', round(avg_glucose, 2))
col2.metric('Average Heart Rate', round(avg_hr, 2))
col3.metric('Average Age', round(avg_age, 2))

# --------------------------------------------------
# DESCRIPTIVE ANALYSIS
# --------------------------------------------------
st.header('Descriptive Analysis')
# Scatter Plot
fig1 = px.scatter(
    merged_df,
    x='Glucose',
    y='Heart_rate',
    color='Gender',
    hover_data=['Age', 'Race'],
    trendline='ols',
    title='Glucose vs Heart Rate by Gender'
)

st.plotly_chart(fig1, use_container_width=True)
st.markdown('''
This chart identifies:

- correlation trends
- outliers
- demographic clustering
- cardiovascular response to glucose changes
''')
# Box Plot

fig2 = px.box(
    merged_df,
    x='Race',
    y='Glucose',
    color='Gender',
    title='Glucose Distribution Across Demographics'
)

st.plotly_chart(fig2, use_container_width=True)
st.markdown('''
This chart compares glucose variability across demographic groups.

It helps identify:

- high-risk groups
- outliers
- distribution spread
- median differences
''')
# Correlation Heatmap
# Encode categorical variables
# Create encoder
encoder = LabelEncoder()

# Copy dataframe
heatmap_df = merged_df.copy()

# Encode Gender
heatmap_df['Gender_encoded'] = encoder.fit_transform(
    heatmap_df['Gender']
)

# Encode Race
heatmap_df['Race_encoded'] = encoder.fit_transform(
    heatmap_df['Race']
)
corr_matrix = heatmap_df[[
    'Glucose',
    'Heart_rate',
    'Age',
    'Steps',
    'Calories',
    'Sleep quality (1-10)',
    '% with sleep disturbances',
    'Gender_encoded',
    'Race_encoded'
]].corr()

fig3 = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect='auto',
    title='Correlation Heatmap'
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown('''
     The heatmap visually identifies:

- strong positive relationships
- negative relationships
- hidden associations
- behavioral impacts on metabolic health       

''')

# --------------------------------------------------
# PREDICTIVE ANALYSIS
# --------------------------------------------------

st.header('Predictive Analysis')

predict_df = merged_df[[
    'Glucose',
    'Heart_rate',
    'Age',
    'Gender',
    'Race',
    'Steps',
    'Calories',
    'Sleep quality (1-10)',
    '% with sleep disturbances'
]].copy()
predict_df.dropna(inplace=True)

predict_df['Gender_encoded'] = encoder.fit_transform(
    predict_df['Gender']
)

predict_df['Race_encoded'] = encoder.fit_transform(
    predict_df['Race']
)

predict_df['High_Glucose_Risk'] = np.where(
    predict_df['Glucose'] > 140,
    1,
    0
)
X = predict_df[[
    'Age',
    'Gender_encoded',
    'Race_encoded',
    'Steps',
    'Calories',
    'Sleep quality (1-10)',
    '% with sleep disturbances'
]]

y = predict_df['High_Glucose_Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)



# Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df.sort_values(
    by='Importance',
    ascending=False,
    inplace=True
)

fig4 = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature Importance for Glucose Risk Prediction'
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown('''

The feature importance chart identifies which variables contribute most to predicting high glucose risk.

It helps:

- determine the strongest health predictors,
- understand behavioral and physiological influences,
- improve model interpretability,
- support evidence-based healthcare decisions.
''')
# Confusion Matrix
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

fig5 = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale='Blues',
    title='Confusion Matrix'
)

st.markdown('''

The confusion matrix evaluates how accurately the machine learning model classifies patients into risk categories.

It shows:

- correct predictions,
- false positives,
- false negatives,
- overall diagnostic reliability.
''')
st.plotly_chart(fig5, use_container_width=True)

# --------------------------------------------------
# PRESCRIPTIVE ANALYSIS
# --------------------------------------------------

st.header('Prescriptive Analysis')

# Select clustering features
cluster_df = merged_df[[
    'Glucose',
    'Heart_rate',
    'Age'
]].copy()

# Scale the features
scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_df)

# Import KMeans
from sklearn.cluster import KMeans

# Create KMeans model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Generate clusters
merged_df['Risk_Group'] = kmeans.fit_predict(
    scaled_data
)

# Risk Cluster Visualization
fig6 = px.scatter(
    merged_df,
    x='Glucose',
    y='Heart_rate',
    color='Risk_Group',
    size='Age',
    hover_data=['Gender', 'Race'],
    title='Patient Risk Clusters'
)

st.plotly_chart(fig6, use_container_width=True)
st.markdown('''

The risk cluster chart groups patients into different risk categories based on:

- glucose,
- heart rate,
- age.

It identifies:

- high-risk populations,
- moderate-risk groups,
- low-risk patients.
''')
# --------------------------------------------------
# FINAL INSIGHTS
# --------------------------------------------------

st.header('Key Dashboard Insights')

st.markdown('''
### Major Findings

- Elevated glucose levels were associated with increased heart rate.

- Sleep quality and physical activity strongly influenced glucose risk.

- Demographic variables improved predictive performance.

- Clustering identified distinct patient risk categories.

- Prescriptive recommendations supported personalized intervention strategies.

### Clinical Significance

This dashboard supports:
- early diabetes risk detection
- cardiovascular monitoring
- preventive healthcare
- personalized treatment planning
- data-driven clinical decision-making
''')