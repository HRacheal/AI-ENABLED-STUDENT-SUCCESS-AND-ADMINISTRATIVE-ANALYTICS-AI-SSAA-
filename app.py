import streamlit as st
import pandas as pd
import joblib


# ==========================================
# 1. LOAD SAVED MODEL AND PREPROCESSORS
# ==========================================

model = joblib.load("logistic_regression_model.pkl")
preprocessor = joblib.load("student_preprocessor.pkl")
label_encoder = joblib.load("target_label_encoder.pkl")


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Student Success Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 3. TITLE
# ==========================================

st.title("🎓 AI Student Dropout & Success Prediction System")

st.write(
    "This application uses machine learning to predict whether a student "
    "is likely to Dropout, remain Enrolled, or Graduate."
)

st.divider()


# ==========================================
# 4. STUDENT INFORMATION
# ==========================================

st.header("Enter Student Information")


# ------------------------------------------
# Demographic Information
# ------------------------------------------

st.subheader("Demographic Information")

col1, col2, col3 = st.columns(3)

with col1:
    marital_status = st.number_input(
        "Marital Status",
        min_value=1,
        value=1
    )

with col2:
    nationality = st.number_input(
        "Nationality",
        min_value=1,
        value=1
    )

with col3:
    gender = st.number_input(
        "Gender",
        min_value=0,
        max_value=1,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    displaced = st.number_input(
        "Displaced",
        min_value=0,
        max_value=1,
        value=0
    )

with col2:
    international = st.number_input(
        "International",
        min_value=0,
        max_value=1,
        value=0
    )

with col3:
    educational_special_needs = st.number_input(
        "Educational Special Needs",
        min_value=0,
        max_value=1,
        value=0
    )


# ==========================================
# 5. ACADEMIC INFORMATION
# ==========================================

st.subheader("Academic Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    application_mode = st.number_input(
        "Application Mode",
        min_value=1,
        value=1
    )

with col2:
    application_order = st.number_input(
        "Application Order (Preference Choice)",
        min_value=0,
        max_value=10,
        value=1
    )

with col3:
    course = st.number_input(
        "Course",
        min_value=1,
        value=1
    )

with col4:
    daytime_attendance = st.number_input(
        "Daytime/Evening Attendance",
        min_value=0,
        max_value=1,
        value=1
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    previous_qualification = st.number_input(
        "Previous Qualification",
        min_value=1,
        value=1
    )

with col2:
    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

with col3:
    admission_grade = st.number_input(
        "Admission Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

with col4:
    age_at_enrollment = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=100,
        value=20
    )


# ==========================================
# 6. FAMILY / BACKGROUND INFORMATION
# ==========================================

st.subheader("Family and Background Information")

col1, col2, col3 = st.columns(3)

with col1:
    mothers_qualification = st.number_input(
        "Mother's Qualification",
        min_value=1,
        value=1
    )

with col2:
    fathers_qualification = st.number_input(
        "Father's Qualification",
        min_value=1,
        value=1
    )

with col3:
    mothers_occupation = st.number_input(
        "Mother's Occupation",
        min_value=1,
        value=1
    )


col1, col2 = st.columns(2)

with col1:
    fathers_occupation = st.number_input(
        "Father's Occupation",
        min_value=1,
        value=1
    )

with col2:
    scholarship_holder = st.number_input(
        "Scholarship Holder",
        min_value=0,
        max_value=1,
        value=0
    )


# ==========================================
# 7. FINANCIAL & ECONOMIC INFORMATION
# ==========================================

st.subheader("Financial & Economic Information")

col1, col2 = st.columns(2)

with col1:
    debtor = st.number_input(
        "Debtor",
        min_value=0,
        max_value=1,
        value=0
    )

with col2:
    tuition_fees_up_to_date = st.number_input(
        "Tuition Fees Up to Date",
        min_value=0,
        max_value=1,
        value=1
    )

col1, col2, col3 = st.columns(3)

with col1:
    unemployment_rate = st.number_input(
        "Unemployment Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.8
    )

with col2:
    inflation_rate = st.number_input(
        "Inflation Rate (%)",
        min_value=-10.0,
        max_value=100.0,
        value=1.4
    )

with col3:
    gdp = st.number_input(
        "GDP Indicator",
        min_value=-50.0,
        max_value=50.0,
        value=1.74
    )


# ==========================================
# 8. SEMESTER PERFORMANCE
# ==========================================

st.subheader("Academic Performance")

col1, col2, col3 = st.columns(3)

with col1:
    curricular_units_1st_credited = st.number_input(
        "1st Semester Units Credited",
        min_value=0,
        value=0
    )

with col2:
    curricular_units_1st_enrolled = st.number_input(
        "1st Semester Units Enrolled",
        min_value=0,
        value=0
    )

with col3:
    curricular_units_1st_evaluations = st.number_input(
        "1st Semester Units Evaluated",
        min_value=0,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    curricular_units_1st_approved = st.number_input(
        "1st Semester Units Approved",
        min_value=0,
        value=0
    )

with col2:
    curricular_units_1st_grade = st.number_input(
        "1st Semester Grade",
        min_value=0.0,
        max_value=20.0,
        value=10.0
    )

with col3:
    curricular_units_1st_without_evaluations = st.number_input(
        "1st Semester Units Without Evaluations",
        min_value=0,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    curricular_units_2nd_credited = st.number_input(
        "2nd Semester Units Credited",
        min_value=0,
        value=0
    )

with col2:
    curricular_units_2nd_enrolled = st.number_input(
        "2nd Semester Units Enrolled",
        min_value=0,
        value=0
    )

with col3:
    curricular_units_2nd_evaluations = st.number_input(
        "2nd Semester Units Evaluated",
        min_value=0,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    curricular_units_2nd_approved = st.number_input(
        "2nd Semester Units Approved",
        min_value=0,
        value=0
    )

with col2:
    curricular_units_2nd_grade = st.number_input(
        "2nd Semester Grade",
        min_value=0.0,
        max_value=20.0,
        value=10.0
    )

with col3:
    curricular_units_2nd_without_evaluations = st.number_input(
        "2nd Semester Units Without Evaluations",
        min_value=0,
        value=0
    )


# ==========================================
# 9. PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Student Outcome",
    type="primary"
)


# ==========================================
# 10. MAKE PREDICTION
# ==========================================

if predict_button:

    student_data = pd.DataFrame({
        "Marital Status": [marital_status],
        "Application mode": [application_mode],
        "Application order": [application_order],
        "Course": [course],
        "Daytime/evening attendance": [daytime_attendance],
        "Previous qualification": [previous_qualification],
        "Previous qualification (grade)": [previous_qualification_grade],
        "Nacionality": [nationality],
        "Mother's qualification": [mothers_qualification],
        "Father's qualification": [fathers_qualification],
        "Mother's occupation": [mothers_occupation],
        "Father's occupation": [fathers_occupation],
        "Displaced": [displaced],
        "Educational special needs": [educational_special_needs],
        "Debtor": [debtor],
        "Tuition fees up to date": [tuition_fees_up_to_date],
        "Gender": [gender],
        "Scholarship holder": [scholarship_holder],
        "Age at enrollment": [age_at_enrollment],
        "International": [international],
        "Admission grade": [admission_grade],
        "Curricular units 1st sem (credited)": [
            curricular_units_1st_credited
        ],
        "Curricular units 1st sem (enrolled)": [
            curricular_units_1st_enrolled
        ],
        "Curricular units 1st sem (evaluations)": [
            curricular_units_1st_evaluations
        ],
        "Curricular units 1st sem (approved)": [
            curricular_units_1st_approved
        ],
        "Curricular units 1st sem (grade)": [
            curricular_units_1st_grade
        ],
        "Curricular units 1st sem (without evaluations)": [
            curricular_units_1st_without_evaluations
        ],
        "Curricular units 2nd sem (credited)": [
            curricular_units_2nd_credited
        ],
        "Curricular units 2nd sem (enrolled)": [
            curricular_units_2nd_enrolled
        ],
        "Curricular units 2nd sem (evaluations)": [
            curricular_units_2nd_evaluations
        ],
        "Curricular units 2nd sem (approved)": [
            curricular_units_2nd_approved
        ],
        "Curricular units 2nd sem (grade)": [
            curricular_units_2nd_grade
        ],
        "Curricular units 2nd sem (without evaluations)": [
            curricular_units_2nd_without_evaluations
        ],
        "Unemployment rate": [unemployment_rate],
        "Inflation rate": [inflation_rate],
        "GDP": [gdp]
    })


    # --------------------------------------
    # Preprocess input
    # --------------------------------------

    student_processed = preprocessor.transform(student_data)


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(student_processed)

    prediction_probability = model.predict_proba(
        student_processed
    )


    predicted_class = label_encoder.inverse_transform(
        prediction
    )[0]


    # ======================================
    # 11. DISPLAY RESULT
    # ======================================

    st.subheader("Prediction Result")

    if predicted_class == "Dropout":
        st.error(
            f"Predicted Outcome: **{predicted_class}**"
        )

    elif predicted_class == "Enrolled":
        st.warning(
            f"Predicted Outcome: **{predicted_class}**"
        )

    else:
        st.success(
            f"Predicted Outcome: **{predicted_class}**"
        )


    # ======================================
    # 12. DISPLAY PROBABILITIES
    # ======================================

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame(
        prediction_probability,
        columns=label_encoder.classes_
    )

    probability_df = probability_df.T.reset_index()

    probability_df.columns = [
        "Outcome",
        "Probability"
    ]

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    display_df = probability_df.copy()

    display_df["Probability"] = (
        display_df["Probability"].astype(str) + "%"
    )

    st.table(display_df)


    # ======================================
    # 13. INTERPRETATION
    # ======================================

    st.subheader("Interpretation")

    highest_probability = probability_df.loc[
        probability_df["Probability"].idxmax()
    ]

    st.write(
        f"The model predicts **{predicted_class}** "
        f"as the most likely student outcome."
    )

    st.write(
        f"The highest predicted probability is "
        f"**{highest_probability['Probability']}%**."
    )

    st.info(
        "This prediction is generated by a machine learning model "
        "and should support, not replace, professional academic "
        "advising and decision-making."
    )