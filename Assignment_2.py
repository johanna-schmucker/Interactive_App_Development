#Individual Assignment 2

#Necessary Imports
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc)
import joblib
import io


#Application Title
st.title("Interactive Machine Learning App")

#Session State in order not to lose a trained model as soon as an interaction (f.e: change a model parameter) happened.
    #Saves if a model was previously trained or not
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    #Saves the trained model for later accessibility
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
    #Saves all the previously tested data
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

#To prevent constant re-runs of code. Functions only run once, download the data and then the result is reused.
@st.cache_data
def load_data(dataset_name):
    #For the Seaborn Datasets
    return sns.load_dataset(dataset_name)

@st.cache_data
def read_csv(file):
    #For an original Dataset
    return pd.read_csv(file)


#Selection of the Dataset
st.header("Load Your Dataset")
#User can choose between seaborn sample datasets or upload their own file (Radio Option)
data_option = st.radio("Select a data source:", ["Use Seaborn dataset", "Upload CSV"])
#If the user selects his own CSV file
if data_option == "Upload CSV":
    #Upload a file locally of a CSV type
    file = st.file_uploader("Upload your CSV file", type = ["csv"])
    if file:
        df = read_csv(file)
        st.write("File was successfully uploaded")
    else:
        st.write("Please upload a CSV file to continue")
        st.stop()
#If the user selects any Seaborn CSV file using a checkbox which includes all seaborn CSV files by name
else:
    dataset_name = st.selectbox("Choose a Seaborn dataset: ", sns.get_dataset_names())
    df = sns.load_dataset(dataset_name)
    st.write(f"Loaded Seaborn dataset: {dataset_name}")

#Show the first few rows of the selected dataset
st.write("Preview of your data: ")
st.dataframe(df.head())


#Data Preparation and Manipulation
st.header("Data Cleaning of the selected Dataset")

#Function to remove outliers of a dataset that could be misleading
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df

#Function which imputes data for missing values using the median for numeric variables and the mode for categorical variables
def fill_missing_values(df):
    for column in df.columns:
        #If the variable that has missing values is numeric
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = df[column].fillna(df[column].median())
        #If the variable that has missing values is categorical
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
    return df

#The user can activate the previously defined automatic data cleaning or decide to not clean data
clean_data = st.checkbox("Clean data: Remove outliers and fill missing values", value = True)
#If the user decides to use the automatic data cleaning, then call the functions and apply it to the dataset
if clean_data:
    numeric_columns = list(df.select_dtypes(include = "number").columns)
    df = fill_missing_values(df)
    df = remove_outliers(df, numeric_columns)
    st.write("Missing values filled and outliers removed")

#Data Preparation
st.header("Select Features and Target")
#Get all column names
columns = list(df.columns)
#Separate numeric columns
numeric_columns = list(df.select_dtypes(include = "number").columns)
categorical_columns = list(df.select_dtypes(include = "object").columns)


#Features and Target Variable
#Choose the target variable
target = st.selectbox("Select the target variable (label):", columns)

#Select the Input Features
numeric_features = []
for x in numeric_columns:
    if x != target:
        numeric_features.append(x)
#Remove the target from categorical features
categorical_features = []
for y in categorical_columns:
    if y != target:
        categorical_features.append(y)

#Choose input features (independent variables) by creating dropdown lists for categorical and numerical features
st.subheader("Choose the input features")
selected_numeric = st.multiselect("Numeric features:", numeric_features)
selected_categorical = st.multiselect("Categorical features:", categorical_features)

#Make sure the user picked at least one feature
if not selected_numeric and not selected_categorical:
    st.warning("Please select at least one feature.")
    st.stop()

#Combines the selected numeric and categorical features as well as creates a new dataset with just the inputs and the target (copy version)
features = selected_numeric + selected_categorical
df_model = df[features + [target]].copy()

#Convert selected categorical columns to numeric ones using one-hot encoding convertion mechanism
df_model = pd.get_dummies(df_model, columns = selected_categorical, drop_first = True)

#Split final prepared features and target variables
X = df_model.drop(columns = [target])
y = df_model[target]

#Normalise the features of the input matrix for the model (important for KNN)
scaler = StandardScaler()
#Apply the scaler to all columns and replace the newly obtained values with the scaled values
X[X.columns] = scaler.fit_transform(X[X.columns])

#Warning if Dataset has < 50 rows
if len(df_model) < 50:
    st.warning("This dataset is quite small. Results may not generalize well.")
#Warning if the Classification target has too many classes or the classes are very imbalanced
if y.nunique() > 20 and y.dtype != "float":
    st.warning("This looks like a classification task with many classes.")


#Machine Learning Models
#Choose Model
st.header("Choose Your Model")

#If the target has few unique values, assume it is classification
if y.nunique() <= 4 and y.dtype != "float":
    task_type = "Classification"
else:
    task_type = "Regression"

#Show model choices depending on task type
if task_type == "Regression":
    #Let the user select one out of three possible regression models
    model_name = st.selectbox("Choose a Regression model:", ["Linear Regression", "Random Forest Regressor", "KNN Regressor"])
else:
    #Let the user select one out of three possible classification models
    model_name = st.selectbox("Choose a Classification model:", ["Logistic Regression", "Random Forest Classifier", "KNN Classifier"])


#Specific Model Parameter Configurations
with st.form("model_form"):
    st.subheader("Model Parameters")
    #Split the original dataset for testing
    test_size = st.slider("Choose test size (portion for testing):", 0.1, 1.0, 0.1)
    #Extra Parameters for a Random Forest such as the number of Trees and the depth of each Tree
    if "Random Forest" in model_name:
        trees = st.slider("Number of trees (Random Forest only):", 10, 200, 100)
        depth = st.slider("Max tree depth:", 1, 20, 5)
    #Extra Parameters for a KNN model such as the number specified of neighbors
    elif "KNN" in model_name:
        k_value = st.slider("Number of neighbors (K for KNN):", 1, 20, 5)
    #If the User submits these Parameters, the model is ready to be trained
    submit = st.form_submit_button("Train Model")


#Train and Test the selected Machine Learning Model
#Train and Evaluate
st.header("Train and Evaluate Model")
if submit:
    #If classification and target is text, convert it to numbers because most Machine Learning Methods can only handle numeric target variables
    if task_type == "Classification" and not np.issubdtype(y.dtype, np.integer):
        y, class_names = pd.factorize(y)
        st.write("Target labels encoded as:")
        for i, label in enumerate(class_names):
            st.write(f"{i}: {label}")
    #If the model used is a Regression model
    elif task_type == "Regression":
        st.write(f"The model will predict continuous numeric values of: `{target}`")

    #Split data according to the testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)


    #Initialize the model based on the user's selection
    #Regression Models
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators = trees, max_depth = depth)
    elif model_name == "KNN Regressor":
        model = KNeighborsRegressor(n_neighbors = k_value)
    #Classification Models
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter = 1000)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators = trees, max_depth = depth)
    elif model_name == "KNN Classifier":
        model = KNeighborsClassifier(n_neighbors = k_value)

    #Train the selected model according to the test/train split
    #Train the model
    model.fit(X_train, y_train)
    #Test the model
    y_pred = model.predict(X_test)

    #Save to session state
    st.session_state.model_trained = True
    st.session_state.trained_model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    
    #Show results of the trained and tested model
    st.header("Model Results")

    #For Regression Models
    if task_type == "Regression":
        #Print out metrics (R-Squared Score and Mean Square Error) with two decimals
        st.write(f"R^2 Score: **{r2_score(y_test, y_pred):.2f}**")
        st.write(f"Mean Squared Error: **{mean_squared_error(y_test, y_pred):.2f}**")

        #Simple Error Plot
        st.subheader("Residual Error Plot")
        #Get the actual Error
        errors = y_test - y_pred
        #Start setting parameters for Plotting
        figure_1, axes = plt.subplots()
        axes.hist(errors, bins = 30)
        axes.set_title("Residual Errors for Regression Models")
        axes.set_xlabel("Error")
        axes.set_ylabel("Frequency")
        st.pyplot(figure_1)

        #Plot that shows the Predicted, Actual value as well as the Residual Error
        st.subheader("Residuals: Bar and Line Overlay")
        #Get the actual Error
        residuals = y_test - y_pred
        df_plot = pd.DataFrame({"Predicted": y_pred, "Actual": y_test, "Residual": residuals}).sort_values(by = "Predicted").reset_index(drop = True)
        #Start setting parameters for Plotting
        figure_2, axes = plt.subplots(figsize = (8, 5))
        # 1. Bar Plot: shows the Residuals/Errors
        axes.bar(df_plot.index, df_plot["Residual"], label = "Residual", color = "skyblue", alpha = 0.7)
        # 2. Line Plots: one for the Actual and Predicted values
        axes.plot(df_plot.index, df_plot["Actual"], label = "Actual", color = "green", linestyle = "--", marker = 'o')
        axes.plot(df_plot.index, df_plot["Predicted"], label = "Predicted", color = "blue", linestyle = "-", marker = 'x')
        #Show the Zero line
        axes.axhline(0, color = "black", linestyle = ":", linewidth = 1)
        axes.set_title("Residuals with Actual vs. Predicted Overlay", fontsize = 14)
        axes.set_xlabel("Sorted Prediction Index", fontsize = 12)
        axes.set_ylabel("Residual/Error Value", fontsize = 12)
        axes.grid(True, linestyle = "--", alpha = 0.3)
        axes.legend()

        # Show in Streamlit
        st.pyplot(figure_2)


    #For Classification Models
    else:
        #Print out metrics (Confusion Matrix, Classification Report, ROC Curve)
        #Confusion Matrix
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        figure_3, axes = plt.subplots()
        sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", ax = axes)
        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")
        st.pyplot(figure_3)
        #Classification Report
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        #ROC Curve (for binary classification only)
        #hasattr checks whether a given object has a specific attribute
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            y_probability = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probability)
            roc_auc = auc(fpr, tpr)
            st.subheader("ROC Curve")
            #Start setting parameters for Plotting
            figure_4, axes = plt.subplots()
            axes.plot(fpr, tpr, label = f"AUC = {roc_auc:.2f}")
            axes.plot([0, 1], [0, 1], linestyle = "--")
            axes.set_xlabel("False Positive Rate")
            axes.set_ylabel("True Positive Rate")
            axes.set_title("ROC Curve")
            axes.legend()
            st.pyplot(figure_4)


    
    #Feature Importance with Permutation (works for all models)
    st.subheader("Feature Importance")

    #Use permutation method to check feature importance (in the end because it needs a trained model and test data predictions to evaluate how "important" each feature really is; shuffle 10 times)
    permutation = permutation_importance(model, X_test, y_test, n_repeats = 10, random_state = 42)
    importances = permutation.importances_mean
    sorted_idx = np.argsort(importances)[::-1]
    #Displays the importance score
    for index in sorted_idx:
        st.write(f"{X.columns[index]}: {importances[index]:.2f}")
    # 1. Plot Feature Importances with a simple Barplot
    st.subheader("Feature Importance Barplot")
    figure_5, axes = plt.subplots()
    sorted_features = X.columns[sorted_idx]
    sns.barplot(x = importances[sorted_idx], y = sorted_features, ax = axes)
    axes.set_title("Feature Importance (Permutation)")
    axes.set_xlabel("Mean Importance")
    axes.set_ylabel("Features")
    st.pyplot(figure_5)

    # 2. Plot feature Importances with a combined Bar and Pareto Plot
    st.subheader("Feature Importance: Pareto Style")

    # 2. Feature Importance Plot
    #get the right data to visualize the plot accordingly
    sorted_features = X.columns[sorted_idx]
    sorted_importance = importances[sorted_idx]
    cumulative = np.cumsum(sorted_importance) / np.sum(sorted_importance)
    #Start setting the parameters for plotting
    figure_5, axes1 = plt.subplots(figsize = (8, 5))
    #Subplot 1: Bar plot
    axes1.bar(sorted_features, sorted_importance, color = "skyblue", label = "Importance")
    #Graph Labels
    axes1.set_ylabel("Mean Importance")
    axes1.set_xlabel("Features")
    axes1.tick_params(axis = 'x', rotation = 45)
    #Subplot 2: Line plot for Cumulative Importance
    axes2 = axes1.twinx()
    axes2.plot(sorted_features, cumulative, color = "red", marker = "o", label = "Cumulative")
    #Graph Labels
    axes2.set_ylabel("Cumulative Importance")
    axes2.axhline(0.8, color = "gray", linestyle = "--", alpha = 0.5)
    axes2.set_ylim([0, 1.05])

    #Plotting the combined Graphs
    figure_5.suptitle("Feature Importance with Cumulative Contribution", fontsize = 14)
    figure_5.legend(loc = "upper right")
    st.pyplot(figure_5)


    #Model-Export as .pkl file
    st.subheader("Export Trained Model")
    #Only if the model was already trained and only if the trained model is saved st.session_state.trained_model it is allowed to be exported
    if st.session_state.model_trained and st.session_state.trained_model:
        #Use two Python libraries (IO and Joblib)
        model_buffer = io.BytesIO()
        joblib.dump(st.session_state.trained_model, model_buffer)
        model_buffer.seek(0)
        #Shows a "Download" button in the Streamlit app
        st.download_button(label = "Download Model", data = model_buffer, file_name = "model.pkl", mime = "application/octet-stream")
    #If no model has been trained yet, tell the User
    else:
        st.write("Train a model first to enable download.")
