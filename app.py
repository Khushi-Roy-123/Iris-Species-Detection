import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris # Keeping this comment to avoid complex diff, but unused now. actually remove it.
# Unused import removed

# Set page config
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
    }
    .stSidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1 {
        color: #ff4b4b;
    }
    .css-1aumxhk {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv('Iris.csv')
# df['species'] is already in string format like 'Iris-setosa'

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", use_column_width=True)
st.sidebar.title("🌸 Iris Classifier")
st.sidebar.markdown("Tune parameters & predict!")

model_name = st.sidebar.selectbox(
    "Select Model",
    ("Random Forest", "Logistic Regression", "KNN", "Decision Tree")
)

st.sidebar.subheader("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['SepalLengthCm'].min()), float(df['SepalLengthCm'].max()), float(df['SepalLengthCm'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['SepalWidthCm'].min()), float(df['SepalWidthCm'].max()), float(df['SepalWidthCm'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['PetalLengthCm'].min()), float(df['PetalLengthCm'].max()), float(df['PetalLengthCm'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['PetalWidthCm'].min()), float(df['PetalWidthCm'].max()), float(df['PetalWidthCm'].mean()))

# Main App
st.title("Iris Flower Classification Web App")
st.markdown("""
This app predicts the **Iris flower species** based on sepal and petal measurements.
Explore the dataset correlations below!
""")

# Load Model
model_file_map = {
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "KNN": "knn.pkl",
    "Decision Tree": "decision_tree.pkl"
}

try:
    model = joblib.load(f"models/{model_file_map[model_name]}")
except FileNotFoundError:
    st.error("Model file not found. Please train models first.")
    st.stop()

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader("Prediction Result")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Predicted Species:**")
    # Prediction is directly the string label now
    st.success(f"🌿 {prediction[0]}")

with col2:
    st.markdown(f"**Confidence Scores:**")
    # Model classes_ attribute has the class names
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.dataframe(proba_df.style.highlight_max(axis=1))

# Visualizations
st.divider()
st.subheader("Data Visualization")

viz_type = st.radio("Choose Visualization", ["Scatter Plot", "Pairplot"])

feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

if viz_type == "Scatter Plot":
    col_x = st.selectbox("X-axis Feature", feature_cols, index=0)
    col_y = st.selectbox("Y-axis Feature", feature_cols, index=1)
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col_x, y=col_y, hue="Species", palette="deep", ax=ax)
    
    # User point
    ax.scatter(input_data[0][feature_cols.index(col_x)], 
               input_data[0][feature_cols.index(col_y)], 
               color='red', s=100, marker='*', label='Your Input')
    ax.legend()
    st.pyplot(fig)

elif viz_type == "Pairplot":
    st.markdown("Generating pairplot... this might take a second.")
    fig = sns.pairplot(df, hue="Species", palette="deep")
    st.pyplot(fig)

st.write("---")
st.markdown("Built with ❤️ using Streamlit")
