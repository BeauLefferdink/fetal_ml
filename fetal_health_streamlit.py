# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# make titles 
st.title('Fetal Health Classification: A Machine Learning App') 
st.image("fetal_health_image.gif")
st.write("Utilize our advanced Mchaine Learning application to predict fetal health classifications.")

         

# import CSVs
default_df = pd.read_csv('fetal_health.csv')
sample_df = pd.read_csv('fetal_health_user.csv')

# Import pickles
model_files = {
    "Decision Tree": "decision_tree.pickle",
    "Random Forest": "random_forest.pickle",
    "AdaBoost": "adaboost.pickle", 
    "Voting Classifier": "soft_voting.pickle"
}

models = {}
for model_name, file_name in model_files.items():
    with open(file_name, 'rb') as model_file:
        models[model_name] = pickle.load(model_file)


# Sidebar
st.sidebar.header("Fetal Health Features Input")
st.sidebar.write('Upload Your Data')
uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])

st.sidebar.warning('Ensure your data strictly follows the format outlined below.', icon="⚠️")
st.sidebar.dataframe(sample_df)
selected_model_name = st.sidebar.radio("Choose Model For Prediction", ["Random Forest", "Decision Tree", "AdaBoost", "Voting Classifier"])
selected_model = models[selected_model_name]
st.sidebar.info(f"🔵 You selected:  **{selected_model_name}**.")


predicted_classes = []
predicted_probabilities = []

if uploaded_file is not None:
    st.success("✅ CSV file uploaded successfully.")
    user_data = pd.read_csv(uploaded_file)

    # Class mapping (e.g., Normal, Suspect, Pathological) for 1, 2, and 3
    class_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}

    for index, row in user_data.iterrows():
        # Use the selected model for predictions
        prediction = selected_model.predict(row.to_frame().T)[0]  # Predicted class
        probabilities = selected_model.predict_proba(row.to_frame().T)[0]  # Probabilities for all classes

        # Get the highest probability and its corresponding class
        max_prob_index = probabilities.argmax()  # Index of the highest probability
        max_prob_class = class_mapping[max_prob_index + 1]  # Map index to class (1-based)
        max_prob_value = probabilities[max_prob_index] *100 # Highest probability value

        # Store the predicted class and highest probability
        predicted_classes.append(class_mapping[prediction])  # Map predicted class
        predicted_probabilities.append(f" {max_prob_value:.1f}")  # Class and probability

    # Display the results
    st.header(f"Predicting Fetal Health Class Using **{selected_model_name}**.")
    results_df = pd.DataFrame({
        "Predicted Class": predicted_classes,
        "Prediction Probability": predicted_probabilities  # Only the highest probability
    })

    # Append results_df to user_data
    final_df = pd.concat([user_data.reset_index(drop=True), results_df], axis=1)


    def highlight_class(val):
        """Apply color based on the Predicted Class."""
        if val == "Normal":
            return "background-color: lime"
        elif val == "Suspect":
            return "background-color: yellow"
        elif val == "Pathological":
            return "background-color: orange"
        return ""

    # Apply the styling
    styled_df = final_df.style.applymap(highlight_class, subset=["Predicted Class"])
    # Display the combined DataFrame
    st.write(styled_df)



    st.subheader("Model Insights")  

    # Define file mapping for model insights
    insight_files = {
        "Decision Tree": {
            "Confusion Matrix": "confusion_mat_dtc.svg",
            "Classification Report": "class_report_dtc.csv",
            "Feature Importance": "feature_imp_dtc.svg"
        },
        "Random Forest": {
            "Confusion Matrix": "confusion_mat_rfc.svg",
            "Classification Report": "class_report_rfc.csv",
            "Feature Importance": "feature_imp_rfc.svg"
        },
        "AdaBoost": {
            "Confusion Matrix": "confusion_mat_abc.svg",
            "Classification Report": "class_report_abc.csv",
            "Feature Importance": "feature_imp_abc.svg"
        },
        "Voting Classifier": {
            "Confusion Matrix": "confusion_mat_svc.svg",
            "Classification Report": "class_report_svc.csv",
            "Feature Importance": "feature_imp_svc.svg"
            }
        }

    # Load the appropriate files based on the selected model
    selected_insight_files = insight_files[selected_model_name]

    # Create tabs for insights
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

    # Tab 1: Confusion Matrix
    with tab1:
        st.write("### Confusion Matrix")
        st.image(selected_insight_files["Confusion Matrix"])


    # Tab 2: Classification Report
    with tab2:
        st.write("### Classification Report")
        # Read and display the classification report
        classification_report_path = selected_insight_files["Classification Report"]
        classification_report_df = pd.read_csv(classification_report_path)
        st.dataframe(classification_report_df)

    # Tab 3: Feature Importance
    with tab3:
        st.write("### Feature Importance")
        st.image(selected_insight_files["Feature Importance"])





else: 
    st.info("ℹ️ *Please upload data to proceed.*")

