import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Feature order used in the original notebook
# ============================================================

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ============================================================
# Categories from the original Telco dataset
#
# These are explicitly supplied so that the encoder produces
# the same one-hot feature structure as the notebook.
# ============================================================

CATEGORIES = [

    # gender
    [
        "Female",
        "Male"
    ],

    # Partner
    [
        "No",
        "Yes"
    ],

    # Dependents
    [
        "No",
        "Yes"
    ],

    # PhoneService
    [
        "No",
        "Yes"
    ],

    # MultipleLines
    [
        "No",
        "No phone service",
        "Yes"
    ],

    # InternetService
    [
        "DSL",
        "Fiber optic",
        "No"
    ],

    # OnlineSecurity
    [
        "No",
        "No internet service",
        "Yes"
    ],

    # OnlineBackup
    [
        "No",
        "No internet service",
        "Yes"
    ],

    # DeviceProtection
    [
        "No",
        "No internet service",
        "Yes"
    ],

    # TechSupport
    [
        "No",
        "No internet service",
        "Yes"
    ],

    # StreamingTV
    [
        "No",
        "No internet service",
        "Yes"
    ],

    # StreamingMovies
    [
        "No",
        "No internet service",
        "Yes"
    ],

    # Contract
    [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    # PaperlessBilling
    [
        "No",
        "Yes"
    ],

    # PaymentMethod
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
]


# ============================================================
# Create encoder
# ============================================================

encoder = OneHotEncoder(
    categories=CATEGORIES,
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)


# ============================================================
# Fit encoder
#
# We create one artificial row containing every "first"
# category. Because categories are explicitly specified above,
# sklearn knows the complete category structure.
# ============================================================

encoder_fit_data = pd.DataFrame([

    [
        categories[0]
        for categories in CATEGORIES
    ]

], columns=CATEGORICAL_COLUMNS)


encoder.fit(
    encoder_fit_data[CATEGORICAL_COLUMNS]
)


# ============================================================
# Preprocess customer input
# ============================================================

def preprocess_input(data):

    # --------------------------------------------------------
    # Convert JSON input into DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([data])


    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        NUMERIC_COLUMNS +
        CATEGORICAL_COLUMNS
    )


    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing_columns)
        )


    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Validate numerical values
    # --------------------------------------------------------

    if df[NUMERIC_COLUMNS].isnull().any().any():

        raise ValueError(
            "One or more numeric fields contain "
            "invalid values."
        )


    # --------------------------------------------------------
    # Numeric features
    #
    # Same order as:
    #
    # X_train[num_cols].values
    # --------------------------------------------------------

    X_numeric = df[
        NUMERIC_COLUMNS
    ].values


    # --------------------------------------------------------
    # Categorical features
    #
    # Same order as:
    #
    # X_train[cat_cols]
    # --------------------------------------------------------

    X_categorical = encoder.transform(
        df[CATEGORICAL_COLUMNS]
    )


    # --------------------------------------------------------
    # Combine features
    #
    # Same as notebook:
    #
    # np.hstack([
    #     X_train[num_cols].values,
    #     X_train_cat
    # ])
    # --------------------------------------------------------

    X_final = np.hstack([
        X_numeric,
        X_categorical
    ])


    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    if X_final.shape[1] != 30:

        raise ValueError(
            f"Feature mismatch: model expects 30 features "
            f"but preprocessing generated "
            f"{X_final.shape[1]} features."
        )


    return X_final