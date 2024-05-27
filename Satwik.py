import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Function to load and process data
def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                st.error("The file encoding is not supported or the file is empty. Please upload a valid CSV file with UTF-8 or ISO-8859-1 encoding.")
                return None
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a non-empty CSV file.")
            return None
        return df
    return None

def preprocess_data(df):
    st.write("## Dataset Preview")
    st.write(df.head())
    st.write("## Statistical Overview")
    st.write(df.describe())
    st.write("## Missing Values")
    st.write(df.isnull().sum())
    return df

def encode_data(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df.loc[:, column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    return df, label_encoders

def train_models(X, y, task):
    models = {}
    if task == 'Classification':
        models['Logistic Regression'] = LogisticRegression(max_iter=1000)
        models['Decision Tree'] = DecisionTreeClassifier()
        models['Random Forest'] = RandomForestClassifier()
    else:
        models['Linear Regression'] = LinearRegression()
        models['Decision Tree'] = DecisionTreeRegressor()
        models['Random Forest'] = RandomForestRegressor()
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    return trained_models

def evaluate_models(trained_models, X_test, y_test, task):
    evaluations = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if task == 'Classification':
            evaluations[name] = accuracy_score(y_test, y_pred)
        else:
            evaluations[name] = mean_squared_error(y_test, y_pred, squared=False)
    return evaluations

def visualize_data(df):
    st.write("Data Preview:")
    st.dataframe(df.head())
    columns = df.columns.tolist()
    if columns:
        chart_type = st.selectbox("Select chart type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Pie Chart"])
        if chart_type in ["Histogram", "Bar Plot", "Pie Chart"]:
            x_axis = st.selectbox("Select column", columns)
        else:
            x_axis = st.selectbox("Select X-axis column", columns)
            y_axis = st.selectbox("Select Y-axis column", columns)
        plt.figure(figsize=(10, 6))
        if chart_type == "Scatter Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Line Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.lineplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Bar Plot":
            st.write(f"{chart_type} of {x_axis}")
            sns.countplot(data=df, x=x_axis)
        elif chart_type == "Histogram":
            st.write(f"{chart_type} of {x_axis}")
            sns.histplot(df[x_axis], kde=True)
        elif chart_type == "Box Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.boxplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Pie Chart":
            st.write(f"Pie Chart of {x_axis}")
            df[x_axis].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
            plt.ylabel('')
        st.pyplot(plt)
    else:
        st.error("The CSV file does not contain any columns.")

def main():
    fixed_string = "Developer -> Satwik Tiwari\REC BIJNOR(IT)\n from Prayagraj."

    # Display the fixed string at the end of the page
    st.write(fixed_string)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Preprocess", "Visualization", "Model Training", "Predict"])

    if page == "Upload & Preprocess":
        st.title("Upload & Preprocess CSV Data")
        data = load_data()
        if data is not None:
            preprocess_data(data)

    elif page == "Visualization":
        st.title("CSV Data Visualization")
        data = load_data()
        if data is not None:
            visualize_data(data)

    elif page == "Model Training":
        st.title("Train Machine Learning Models")
        data = load_data()
        if data is not None:
            data = preprocess_data(data)
            input_columns = st.multiselect("Select Input Columns", options=data.columns)
            target_column = st.selectbox("Select Target Column", options=data.columns)
            if input_columns and target_column:
                X = data[input_columns]
                y = data[[target_column]]
                X, _ = encode_data(X)
                y, label_encoders = encode_data(y)
                y = y[target_column]
                
                # Check if task type matches the target column
                task = st.radio("Select Task Type", ('Classification', 'Regression'))
                if task == 'Classification' and y.nunique() <= 10:
                    y = LabelEncoder().fit_transform(y)
                elif task == 'Classification' and y.nunique() > 10:
                    st.error("For classification, the target column should have 10 or fewer unique values.")
                    return
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                trained_models = train_models(X_train, y_train, task)
                evaluations = evaluate_models(trained_models, X_test, y_test, task)
                st.write("## Model Performance")
                st.write(evaluations)
                selected_model = st.selectbox("Select Model", options=list(trained_models.keys()))
                st.session_state["selected_model"] = trained_models[selected_model]
                st.session_state["input_columns"] = input_columns
                st.session_state["label_encoder"] = label_encoders.get(target_column, None)

    elif page == "Predict":
        st.title("Make Predictions")
        if "selected_model" in st.session_state:
            model = st.session_state["selected_model"]
            input_columns = st.session_state["input_columns"]
            label_encoder = st.session_state.get("label_encoder", None)
            user_input = {}
            for col in input_columns:
                user_input[col] = st.text_input(f"Input {col}")
            if st.button("Predict"):
                user_input_df = pd.DataFrame([user_input])
                user_input_df, _ = encode_data(user_input_df)
                prediction = model.predict(user_input_df)
                if label_encoder:
                    prediction = label_encoder.inverse_transform(prediction)
                st.write(f"Prediction: {prediction}")
        else:
            st.error("Please train a model first in the 'Model Training' page.")

if __name__ == '__main__':
    main()
