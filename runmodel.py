import base64
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.preprocessing import StandardScaler
# Streamlit app title
st.title('ITI105 Team Project')
st.subheader('Phishing web site Machine Learning Prediction App')

if 'clear_output' not in st.session_state:
    st.session_state.clear_output = False

# Function to clear specific elements
def clear_previous_output():
    st.session_state.clear_output = True

# Load the pre-uploaded dataset
default_file_path = 'https://raw.githubusercontent.com/JimmyYehtut/ITI105Files/main/test_dataset.csv'
df_new = pd.read_csv(default_file_path)


# Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV file with website data", type="csv")
row_index = None
if uploaded_file is not None: 
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    # st.write("Original Dataframe:", df)

    # Extract the URL column to display in the dropdown
    url_list = df['url'].tolist()

    # Display the dropdown with URL options
    selected_url = st.selectbox("Select URL for Prediction", url_list)

    # Display the list fo model
    selected_model = st.selectbox("Select Model for Prediction", ['Random Forest', 'Logistic Regression', 'SVM', 'KNN', 'Decision Tree'])


    # Remove the first (non-numeric) and last (target) columns
    if df.shape[1] > 2:  # Ensure there are enough columns to remove
        features_df = df.iloc[:, 1:-1]  # Drop first and last columns
       

        # Select a row for prediction
        # row_index = st.number_input("Select a row index for prediction", min_value=0, max_value=len(features_df)-1, step=1)
        row_index = df[df['url'] == selected_url].index[0]
        # Display the selected row's features in a table
        selected_row = df.iloc[row_index, :]
        st.subheader("List of selected website features:")
        st.table(selected_row.to_frame().T)
        
    else:
        st.write("The dataset does not have enough columns after removing the first and last columns.")
else:
    # st.error("ERROR!!! Please upload a CSV file to continue.")
    st.write("Using pre-uploaded sample data:")
    df = df_new
    # Extract the URL column to display in the dropdown
    url_list = df['url'].tolist()

    # Display the dropdown with URL options
    selected_url = st.selectbox("Select URL for Prediction", url_list)

    # Display the list fo model
    selected_model = st.selectbox("Select Model for Prediction", ['Random Forest', 'Logistic Regression', 'SVM', 'KNN', 'Decision Tree'])


    # Remove the first (non-numeric) and last (target) columns
    if df.shape[1] > 2:  # Ensure there are enough columns to remove
        features_df = df.iloc[:, 1:-1]  # Drop first and last columns
       

        # Select a row for prediction
        # row_index = st.number_input("Select a row index for prediction", min_value=0, max_value=len(features_df)-1, step=1)
        row_index = df[df['url'] == selected_url].index[0]
        # Display the selected row's features in a table
        selected_row = df.iloc[row_index, :]
        st.subheader("List of selected website features:")
        st.table(selected_row.to_frame().T)
        
    else:
        st.write("The dataset does not have enough columns after removing the first and last columns.")

if st.button("Predict"):

    # Clear previous st.success, st.error, and st.markdown elements
    clear_previous_output()
    file_ = open("It'ok.webp", "rb")
    contents = file_.read()
    data_url_ok = base64.b64encode(contents).decode("utf-8")
    file_.close()

    file = open("Warning.gif", "rb")
    contents = file.read()
    data_url_warning = base64.b64encode(contents).decode("utf-8")
    file.close()
    if row_index is not None:
            input_values = features_df.iloc[row_index].values  # Get selected row data
            # st.write("Selected Features Dataframe for predicton:", input_values)
            # st.write("Selected Row Data (Features Only):", input_values)
            single_sample = np.array(input_values)
            # Dummy model for the purpose of this example
            # Normally you would load a pre-trained model or train one
            # X = features_df  # Using the processed features data
            # y = [0]*len(df)  # Dummy target variable for training the model (since we don't have a real target)

            # # Train/test split
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # model = RandomForestClassifier()
            # model.fit(X_train, y_train)

            # Show progress spinner while making predictions
            with st.spinner('Making prediction...'):
                # Predict based on selected row
                # prediction = model.predict([input_values])
                # Load the pre-trained scaler and model
                with open('scaler.pkl', 'rb') as f:
                    scalar = pickle.load(f)
                
                with open('rf_clf.pkl', 'rb') as f:
                    rf_clf = pickle.load(f)

                            # Scale the new data using the pre-trained scaler
                X_new_scaled = scalar.transform(single_sample.reshape(1, -1))

                # Make predictions using the pre-trained model
                prediction = rf_clf.predict(X_new_scaled)

                # loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))
                # prediction = loaded_model.predict(np.array(single_sample))


            # st.write(f"Prediction : {prediction[0]}")
            if prediction[0] == 0:
                st.success("The website is not a phishing website.")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_ok}" alt="cat gif">', unsafe_allow_html=True,)
            else:
                st.error("The website is a phishing website.")
                st.markdown(f'<img src="data:image/gif;base64,{data_url_warning}" alt="cat gif">', unsafe_allow_html=True,)

            # Note: Since we don't have a real target, accuracy calculation is skipped.
    else:
        st.error("ERROR!!! Please provide web site information for prediction !!!")

# This block clears the elements only if the prediction button is pressed
if st.session_state.clear_output:
    st.session_state.clear_output = False
    # st.success("")  # Clear any previous success messages
    # st.error("")    # Clear any previous error messages
    # st.markdown("") # Clear any previous markdown content
