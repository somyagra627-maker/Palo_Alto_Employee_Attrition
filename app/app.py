import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Risk Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "employee_risk_scores.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_attrition_model.pkl"
)

SHAP_EXPLAINER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "shap_explainer.pkl"
)

SHAP_METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "shap_metadata.pkl"
)

# -----------------------------
# Load Data and Models
# -----------------------------
df = pd.read_csv(DATA_PATH)

model = joblib.load(
    MODEL_PATH
)

shap_explainer = joblib.load(
    SHAP_EXPLAINER_PATH
)

shap_metadata = joblib.load(
    SHAP_METADATA_PATH
)

shap_feature_names = shap_metadata[
    "feature_names"
]

# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Employee Attrition Risk Prediction System"
)

st.markdown(
    """
    **Machine Learning–Based Employee Attrition Prediction and Risk Scoring System**

    This dashboard uses machine learning to estimate employee attrition
    probability and classify employees into Low, Medium, and High Risk categories.
    """
)

st.divider()

# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

total_employees = len(df)

high_risk_count = (
    df["Risk_Category"] == "High Risk"
).sum()

medium_risk_count = (
    df["Risk_Category"] == "Medium Risk"
).sum()

low_risk_count = (
    df["Risk_Category"] == "Low Risk"
).sum()

high_risk_percentage = (
    high_risk_count / total_employees
) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Employees",
    total_employees
)

col2.metric(
    "High Risk Employees",
    high_risk_count
)

col3.metric(
    "Medium Risk Employees",
    medium_risk_count
)

col4.metric(
    "High Risk %",
    f"{high_risk_percentage:.2f}%"
)

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Employee Risk Distribution"
)

risk_distribution = (
    df["Risk_Category"]
    .value_counts()
    .reindex(
        [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]
    )
)

st.bar_chart(
    risk_distribution
)

# ============================================================
# DEPARTMENT-LEVEL RISK
# ============================================================

st.subheader(
    "Department-Level Attrition Risk"
)

department_risk = (
    df.groupby("Department")[
        "Attrition_Probability"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
    * 100
)

st.bar_chart(
    department_risk
)

st.caption(
    "Values represent the average predicted attrition "
    "probability for employees within each department."
)

# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header(
    "🔎 Employee Search"
)

departments = [
    "All Departments"
] + sorted(
    df["Department"]
    .dropna()
    .unique()
    .tolist()
)

selected_department = st.sidebar.selectbox(
    "Select Department",
    departments
)

# Apply department filter
if selected_department == "All Departments":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["Department"] == selected_department
    ].copy()

# Employee selector
employee_options = filtered_df.index.tolist()

if len(employee_options) == 0:

    st.warning(
        "No employees found for the selected department."
    )

else:

    selected_employee_index = st.sidebar.selectbox(
        "Select Employee Record",
        employee_options
    )

    # ========================================================
    # EMPLOYEE RISK PROFILE
    # ========================================================

    st.divider()

    st.subheader(
        "👤 Employee Risk Profile"
    )

    selected_employee = df.loc[
        selected_employee_index
    ]

    employee_probability = (
        selected_employee[
            "Attrition_Probability"
        ]
    )

    employee_risk = (
        selected_employee[
            "Risk_Category"
        ]
    )

    # --------------------------------------------------------
    # Risk summary
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Employee Record",
        selected_employee_index
    )

    col2.metric(
        "Attrition Probability",
        f"{employee_probability * 100:.2f}%"
    )

    col3.metric(
        "Risk Category",
        employee_risk
    )

    # ========================================================
    # EMPLOYEE DETAILS
    # ========================================================

    st.markdown(
        "### Employee Details"
    )

    detail_col1, detail_col2, detail_col3, detail_col4 = (
        st.columns(4)
    )

    detail_col1.write(
        f"**Age:** {selected_employee['Age']}"
    )

    detail_col2.write(
        f"**Job Role:** {selected_employee['JobRole']}"
    )

    detail_col3.write(
        f"**Department:** {selected_employee['Department']}"
    )

    detail_col4.write(
        f"**Monthly Income:** ₹{selected_employee['MonthlyIncome']:,.0f}"
    )

    # ========================================================
    # ADDITIONAL EMPLOYEE DETAILS
    # ========================================================

    detail_col5, detail_col6, detail_col7, detail_col8 = (
        st.columns(4)
    )

    detail_col5.write(
        f"**Job Level:** {selected_employee['JobLevel']}"
    )

    detail_col6.write(
        f"**Job Satisfaction:** {selected_employee['JobSatisfaction']}"
    )

    detail_col7.write(
        f"**Overtime:** {selected_employee['OverTime']}"
    )

    detail_col8.write(
        f"**Years at Company:** {selected_employee['YearsAtCompany']}"
    )

    # ========================================================
    # POTENTIAL CONTRIBUTING FACTORS
    # ========================================================

    st.markdown(
        "### ⚠️ Potential Contributing Factors"
    )

    reasons = []

    if selected_employee["OverTime"] == "Yes":
        reasons.append("Overtime")

    if selected_employee["BusinessTravel"] == "Travel_Frequently":
        reasons.append("Frequent business travel")

    if selected_employee["JobSatisfaction"] <= 2:
        reasons.append("Low job satisfaction")

    if selected_employee["EnvironmentSatisfaction"] <= 2:
        reasons.append("Low environment satisfaction")

    if selected_employee["WorkLifeBalance"] <= 2:
        reasons.append("Low work-life balance")

    if selected_employee["DistanceFromHome"] >= 15:
        reasons.append("Long distance from home")

    if selected_employee["YearsSinceLastPromotion"] >= 3:
        reasons.append("Long time since last promotion")

    if selected_employee["YearsWithCurrManager"] <= 1:
        reasons.append("Short tenure with current manager")

    if selected_employee["NumCompaniesWorked"] >= 4:
        reasons.append("Multiple previous employers")

    if len(reasons) == 0:

        st.success(
            "No major predefined risk indicators "
            "were identified for this employee."
        )

    else:

        for reason in reasons:
            st.write(
                f"• {reason}"
            )

    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 SHAP Explainability"
    )

    st.write(
        "SHAP values show which model features contributed "
        "most strongly to this employee's predicted attrition risk."
    )

    # --------------------------------------------------------
    # Prepare employee data for SHAP
    # --------------------------------------------------------

    employee_features = selected_employee.drop(
        labels=[
            "Attrition",
            "Attrition_Probability",
            "Risk_Category"
        ],
        errors="ignore"
    )

    employee_features_df = pd.DataFrame(
        [employee_features]
    )

    # --------------------------------------------------------
    # Transform employee data
    # --------------------------------------------------------

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    employee_transformed = (
        preprocessor.transform(
            employee_features_df
        )
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    employee_shap_values = (
        shap_explainer.shap_values(
            employee_transformed
        )
    )

    employee_shap_values = np.array(
        employee_shap_values
    ).flatten()

    # --------------------------------------------------------
    # Create SHAP DataFrame
    # --------------------------------------------------------

    employee_shap_df = pd.DataFrame(
        {
            "Feature": shap_feature_names,
            "SHAP_Value": employee_shap_values
        }
    )

    employee_shap_df[
        "Absolute_SHAP"
    ] = employee_shap_df[
        "SHAP_Value"
    ].abs()

    employee_shap_df = (
        employee_shap_df
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
    )

    # ========================================================
    # READABLE FEATURE NAMES
    # ========================================================

    def make_readable_feature_name(feature):

        feature = feature.replace(
            "num__",
            ""
        )

        feature = feature.replace(
            "cat__",
            ""
        )

        feature = feature.replace(
            "_",
            " "
        )

        return feature

    employee_shap_df[
        "Readable_Feature"
    ] = (
        employee_shap_df[
            "Feature"
        ].apply(
            make_readable_feature_name
        )
    )

    # ========================================================
    # RISK-INCREASING FACTORS
    # ========================================================

    risk_increasing = (
        employee_shap_df[
            employee_shap_df["SHAP_Value"] > 0
        ]
        .sort_values(
            "SHAP_Value",
            ascending=False
        )
        .head(5)
    )

    st.markdown(
        "### 📈 Risk-Increasing Factors"
    )

    if len(risk_increasing) > 0:

        st.dataframe(
            risk_increasing[
                [
                    "Readable_Feature",
                    "SHAP_Value"
                ]
            ].rename(
                columns={
                    "Readable_Feature": "Factor",
                    "SHAP_Value": "SHAP Contribution"
                }
            ),
            use_container_width=True
        )

    else:

        st.info(
            "No risk-increasing SHAP factors identified."
        )

    # ========================================================
    # RISK-REDUCING FACTORS
    # ========================================================

    risk_reducing = (
        employee_shap_df[
            employee_shap_df["SHAP_Value"] < 0
        ]
        .sort_values(
            "SHAP_Value",
            ascending=True
        )
        .head(5)
    )

    st.markdown(
        "### 📉 Risk-Reducing Factors"
    )

    if len(risk_reducing) > 0:

        st.dataframe(
            risk_reducing[
                [
                    "Readable_Feature",
                    "SHAP_Value"
                ]
            ].rename(
                columns={
                    "Readable_Feature": "Factor",
                    "SHAP_Value": "SHAP Contribution"
                }
            ),
            use_container_width=True
        )

    else:

        st.info(
            "No risk-reducing SHAP factors identified."
        )

    # ========================================================
    # SHAP INTERPRETATION
    # ========================================================

    st.caption(
        "Interpretation: Positive SHAP values indicate features "
        "that pushed the model prediction toward higher attrition "
        "risk, while negative SHAP values indicate features that "
        "pushed the prediction toward lower attrition risk. "
        "These are model explanations, not causal conclusions."
    )

    # ========================================================
    # WHAT-IF RISK SIMULATION
    # ========================================================

    st.divider()

    st.subheader(
        "🔮 What-If Risk Simulation"
    )

    st.write(
        "Adjust selected employee factors to explore how changes "
        "may affect the model's predicted attrition probability."
    )

    col1, col2 = st.columns(2)

    with col1:

        whatif_job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                selected_employee[
                    "JobSatisfaction"
                ]
            )
        )

        whatif_worklife_balance = st.slider(
            "Work-Life Balance",
            min_value=1,
            max_value=4,
            value=int(
                selected_employee[
                    "WorkLifeBalance"
                ]
            )
        )

    with col2:

        whatif_environment_satisfaction = st.slider(
            "Environment Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                selected_employee[
                    "EnvironmentSatisfaction"
                ]
            )
        )

        whatif_years_promotion = st.slider(
            "Years Since Last Promotion",
            min_value=0,
            max_value=15,
            value=int(
                selected_employee[
                    "YearsSinceLastPromotion"
                ]
            )
        )

    # --------------------------------------------------------
    # Create What-If Employee
    # --------------------------------------------------------

    whatif_employee = selected_employee.copy()

    whatif_employee[
        "JobSatisfaction"
    ] = whatif_job_satisfaction

    whatif_employee[
        "WorkLifeBalance"
    ] = whatif_worklife_balance

    whatif_employee[
        "EnvironmentSatisfaction"
    ] = whatif_environment_satisfaction

    whatif_employee[
        "YearsSinceLastPromotion"
    ] = whatif_years_promotion

    # Recalculate engineered features

    whatif_employee[
        "PromotionDelay"
    ] = int(
        whatif_employee[
            "YearsSinceLastPromotion"
        ] >= 3
    )

    whatif_employee[
        "EngagementScore"
    ] = (
        whatif_employee[
            [
                "JobInvolvement",
                "JobSatisfaction",
                "EnvironmentSatisfaction",
                "RelationshipSatisfaction"
            ]
        ].mean()
    )

    # --------------------------------------------------------
    # Prepare What-If Input
    # --------------------------------------------------------

    whatif_input = pd.DataFrame(
        [whatif_employee]
    )

    # IMPORTANT:
    # Get the exact features expected by the trained model
    model_features = model.named_steps[
        "preprocessor"
    ].feature_names_in_

    whatif_input = whatif_input[
        model_features
    ]

    # --------------------------------------------------------
    # Predict What-If Risk
    # --------------------------------------------------------

    whatif_probability = (
        model.predict_proba(
            whatif_input
        )[0][1]
    )

    # Risk category

    if whatif_probability < 0.30:

        whatif_risk = "Low Risk"

    elif whatif_probability <= 0.60:

        whatif_risk = "Medium Risk"

    else:

        whatif_risk = "High Risk"

    # --------------------------------------------------------
    # Display What-If Result
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Simulated Risk"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Simulated Attrition Probability",
        f"{whatif_probability:.2%}"
    )

    col2.metric(
        "Simulated Risk Category",
        whatif_risk
    )

    # Compare original vs simulated

    risk_difference = (
        whatif_probability
        - employee_probability
    )

    st.metric(
        "Change in Predicted Risk",
        f"{risk_difference:+.2%}"
    )
    # ========================================================
    # HR ACTION RECOMMENDATION
    # ========================================================

    st.divider()

    st.subheader(
        "💡 Suggested HR Action"
    )

    if whatif_risk == "High Risk":

        st.error(
            "Priority attention recommended. "
            "Consider a structured retention discussion, "
            "review of workload, career progression, and "
            "employee engagement."
        )

    elif whatif_risk == "Medium Risk":

        st.warning(
            "Moderate attention recommended. "
            "Consider checking employee satisfaction, "
            "work-life balance, career growth, and manager support."
        )

    else:

        st.success(
            "Current model prediction indicates relatively lower "
            "attrition risk. Continue regular engagement and "
            "retention practices."
        )

    st.caption(
        "These recommendations are decision-support suggestions "
        "based on model-predicted risk and should be combined "
        "with HR judgment and employee context."
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "Note: SHAP values explain the contribution of model "
        "features to the prediction. What-If Simulation shows "
        "how the model prediction changes when selected inputs "
        "are modified. These outputs should not be interpreted "
        "as proof that any individual factor causes employee "
        "attrition."
    )