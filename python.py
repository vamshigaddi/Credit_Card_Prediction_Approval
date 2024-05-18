import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the trained machine learning model
model_file = 'Credit_card_approval_prediction_model.pickle'

with open(model_file, 'rb') as model:
    classifier = pickle.load(model)

# Create a function to predict credit card approval
def predict_credit_approval(features):
    prediction = classifier.predict(features)
    return prediction

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Credit Card Approval Prediction")

# Sidebar with credit card image
# st.sidebar.image('credit_card_image.png', width=200)
# st.image('credit_card_image.png', width=200)

st.header("User Input")

# Collect user input features in the specified order
col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Enter Gender", ["Male", "Female"], key="gender")
    car_owner = st.selectbox("Enter Car Owner", ["Yes", "No"], key="car_owner")
    property_owner = st.selectbox("Enter Property Owner", ["Yes", "No"], key="property_owner")
    type_income = st.selectbox("Enter Type of Income", ['Pensioner', 'Commercial associate', 'Working', 'State servant'], key="type_income")

with col2:
    education = st.selectbox("Enter Education", ['Higher education', 'Secondary / secondary special', 'Lower secondary', 'Incomplete higher', 'Academic degree'], key="education")
    marital_status = st.selectbox("Enter Marital Status", ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'], key="marital_status")
    housing_type = st.selectbox("Enter Housing Type", ['House / apartment', 'With parents', 'Rented apartment', 'Municipal apartment', 'Co-op apartment', 'Office apartment'], key="housing_type")
    work_phone = st.selectbox("Enter Work Phone", ["Yes", "No"], key="work_phone")

with col3:
    phone = st.selectbox("Enter Phone", ["Yes", "No"], key="phone")
    email_id = st.selectbox("Enter Email ID", ["Yes", "No"], key="email_id")
    family_members = st.number_input("Enter Number of Family Members", value=1, key="family_members")

with col4:
    children = st.number_input("Enter Number of Children",format='%d', key="children")
    annual_income = st.number_input("Enter Amount for Annual Income",format='%d', key="annual_income")
    work_experience = st.number_input("Enter Years of Work Experience",format='%d', key="work_experience")
    age = st.number_input("Enter Age", min_value=18, max_value=80, key="age")

# Encode categorical features using Label Encoding
label_encoder = LabelEncoder()
gender = label_encoder.fit_transform([gender])[0]
car_owner = label_encoder.fit_transform([car_owner])[0]
property_owner = label_encoder.fit_transform([property_owner])[0]
work_phone = label_encoder.fit_transform([work_phone])[0]
phone = label_encoder.fit_transform([phone])[0]
email_id = label_encoder.fit_transform([email_id])[0]
type_income = label_encoder.fit_transform([type_income])[0]
education = label_encoder.fit_transform([education])[0]
marital_status = label_encoder.fit_transform([marital_status])[0]
housing_type = label_encoder.fit_transform([housing_type])[0]

# Define the StandardScaler instance
scaler = StandardScaler()

# Preprocess the user input using the loaded or new scaler
annual_income_scaled = scaler.fit_transform(np.array([[annual_income]]))

features = np.array(
    [[gender, car_owner, property_owner, children, annual_income_scaled[0][0], type_income, education, marital_status, housing_type, work_phone, phone, email_id, family_members, work_experience, age]]
)

if st.button("Predict"):
    st.subheader("Prediction")
    prediction = predict_credit_approval(features)
    if prediction[0] == 0:
        st.write("The application is Approved")
    else:
        st.write("The application is Not Approved")

# Styling
st.markdown(
    """
    This is a simple credit card approval prediction app. 
    Enter the required information and click 'Predict' to see the result.
    """
)

