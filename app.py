import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Firewall Rule Recommendation",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

model = joblib.load(
    "models/firewall_model.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

dataset = pd.read_csv(
    "results/processed_firewall_logs.csv"
)


# ==========================================
# TITLE
# ==========================================

st.title(
    "🛡️ AI Firewall Rule Recommendation System"
)

st.write(
    "An AI-based system for predicting firewall actions "
    "and recommending appropriate firewall rules from "
    "network traffic characteristics."
)

st.divider()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Traffic Analysis",
        "Dataset Testing"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("📊 System Dashboard")


    # ------------------------------------------
    # Dataset Overview
    # ------------------------------------------

    st.subheader("Dataset Overview")

    total_records = len(dataset)

    total_actions = dataset["Action"].nunique()

    total_features = 11

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    with col2:

        st.metric(
            "Firewall Actions",
            total_actions
        )

    with col3:

        st.metric(
            "Features",
            total_features
        )


    # ------------------------------------------
    # Model Performance
    # ------------------------------------------

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "99.77%"
        )

    with col2:

        st.metric(
            "Precision",
            "99.76%"
        )

    with col3:

        st.metric(
            "Recall",
            "99.77%"
        )

    with col4:

        st.metric(
            "F1 Score",
            "99.77%"
        )


    # ------------------------------------------
    # Machine Learning Model
    # ------------------------------------------

    st.subheader("Machine Learning Model")

    st.info(
        """
**Selected Model:** Decision Tree

The model was selected based on the evaluation results
obtained during model training.

**Model Accuracy:** 99.77%

The system predicts four firewall actions:

• ALLOW

• DENY

• DROP

• RESET-BOTH
"""
    )


    # ------------------------------------------
    # Firewall Action Distribution
    # ------------------------------------------

    st.subheader(
        "Firewall Action Distribution"
    )

    action_counts = (
        dataset["Action"]
        .value_counts()
        .rename_axis("Action")
        .reset_index(
            name="Number of Records"
        )
    )

    st.bar_chart(
        action_counts.set_index(
            "Action"
        )
    )


    # ------------------------------------------
    # Action Statistics
    # ------------------------------------------

    st.subheader(
        "Action Statistics"
    )

    st.dataframe(
        action_counts,
        use_container_width=True
    )


    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    confusion_matrix_path = (
        "results/confusion_matrix.png"
    )

    if os.path.exists(
        confusion_matrix_path
    ):

        st.image(
            confusion_matrix_path,
            caption=(
                "Firewall Action Classification "
                "Confusion Matrix"
            ),
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image was not found."
        )


# =========================================================
# TRAFFIC ANALYSIS
# =========================================================

elif page == "Traffic Analysis":

    st.header(
        "🔍 AI Traffic Analysis"
    )

    st.write(
        "Enter network traffic information to obtain "
        "an AI-based firewall action prediction."
    )


    # ------------------------------------------
    # Input Fields
    # ------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        source_port = st.number_input(
            "Source Port",
            min_value=0,
            max_value=65535,
            value=57222
        )

        nat_source_port = st.number_input(
            "NAT Source Port",
            min_value=0,
            max_value=65535,
            value=54587
        )

        bytes_value = st.number_input(
            "Bytes",
            min_value=0,
            value=177
        )

        bytes_sent = st.number_input(
            "Bytes Sent",
            min_value=0,
            value=94
        )

        bytes_received = st.number_input(
            "Bytes Received",
            min_value=0,
            value=83
        )

        packets = st.number_input(
            "Packets",
            min_value=0,
            value=2
        )


    with col2:

        destination_port = st.number_input(
            "Destination Port",
            min_value=0,
            max_value=65535,
            value=53
        )

        nat_destination_port = st.number_input(
            "NAT Destination Port",
            min_value=0,
            max_value=65535,
            value=53
        )

        elapsed_time = st.number_input(
            "Elapsed Time (sec)",
            min_value=0,
            value=30
        )

        pkts_sent = st.number_input(
            "Packets Sent",
            min_value=0,
            value=1
        )

        pkts_received = st.number_input(
            "Packets Received",
            min_value=0,
            value=1
        )


    st.divider()


    # ------------------------------------------
    # Analyze Traffic
    # ------------------------------------------

    if st.button(
        "🔎 Analyze Traffic",
        use_container_width=True
    ):

        features = [
            "Source Port",
            "Destination Port",
            "NAT Source Port",
            "NAT Destination Port",
            "Bytes",
            "Bytes Sent",
            "Bytes Received",
            "Packets",
            "Elapsed Time (sec)",
            "pkts_sent",
            "pkts_received"
        ]


        traffic_data = pd.DataFrame(
            [[
                source_port,
                destination_port,
                nat_source_port,
                nat_destination_port,
                bytes_value,
                bytes_sent,
                bytes_received,
                packets,
                elapsed_time,
                pkts_sent,
                pkts_received
            ]],
            columns=features
        )


        # --------------------------------------
        # Prediction
        # --------------------------------------

        prediction = model.predict(
            traffic_data
        )

        predicted_action = (
            label_encoder.inverse_transform(
                prediction
            )[0]
        )


        # --------------------------------------
        # Confidence
        # --------------------------------------

        probabilities = model.predict_proba(
            traffic_data
        )[0]

        confidence = (
            max(probabilities) * 100
        )


        # --------------------------------------
        # AI Prediction
        # --------------------------------------

        st.subheader(
            "🤖 AI Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                "Predicted Firewall Action: "
                f"{predicted_action.upper()}"
            )

        with col2:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


        # --------------------------------------
        # Recommendation
        # --------------------------------------

        if predicted_action == "allow":

            recommendation = "ALLOW"

            reason = (
                "The traffic pattern was classified "
                "as allowed by the machine learning model."
            )

        elif predicted_action == "deny":

            recommendation = "DENY"

            reason = (
                "The traffic pattern was classified "
                "as denied by the machine learning model."
            )

        elif predicted_action == "drop":

            recommendation = "DROP"

            reason = (
                "The traffic pattern was classified "
                "as drop by the machine learning model."
            )

        elif predicted_action == "reset-both":

            recommendation = "RESET CONNECTION"

            reason = (
                "The traffic pattern was classified "
                "as reset-both by the machine learning model."
            )

        else:

            recommendation = "UNKNOWN"

            reason = (
                "Unknown prediction."
            )


        # --------------------------------------
        # Recommended Firewall Rule
        # --------------------------------------

        st.subheader(
            "🛡️ Recommended Firewall Rule"
        )

        st.info(
            f"""
**Action:** {recommendation}

**Source Port:** {source_port}

**Destination Port:** {destination_port}

**Reason:** {reason}
"""
        )


# =========================================================
# DATASET TESTING
# =========================================================

elif page == "Dataset Testing":

    st.header(
        "🧪 Dataset Record Testing"
    )

    st.write(
        "Select an existing firewall traffic record "
        "and compare its actual action with the AI prediction."
    )


    # ------------------------------------------
    # Record Selection
    # ------------------------------------------

    record_number = st.number_input(
        "Dataset Record Number",
        min_value=1,
        max_value=len(dataset),
        value=1,
        step=1
    )


    row_index = (
        int(record_number) - 1
    )

    selected_record = (
        dataset.iloc[row_index]
    )


    # ------------------------------------------
    # Selected Record
    # ------------------------------------------

    st.subheader(
        "Selected Record"
    )

    actual_action = str(
        selected_record["Action"]
    ).lower()


    st.write(
        f"**Record Number:** {record_number}"
    )

    st.write(
        f"**Actual Firewall Action:** "
        f"{actual_action.upper()}"
    )


    # ------------------------------------------
    # Traffic Information
    # ------------------------------------------

    data_display = {

        "Source Port":
            selected_record["Source Port"],

        "Destination Port":
            selected_record["Destination Port"],

        "NAT Source Port":
            selected_record["NAT Source Port"],

        "NAT Destination Port":
            selected_record["NAT Destination Port"],

        "Bytes":
            selected_record["Bytes"],

        "Bytes Sent":
            selected_record["Bytes Sent"],

        "Bytes Received":
            selected_record["Bytes Received"],

        "Packets":
            selected_record["Packets"],

        "Elapsed Time (sec)":
            selected_record["Elapsed Time (sec)"],

        "Packets Sent":
            selected_record["pkts_sent"],

        "Packets Received":
            selected_record["pkts_received"]
    }


    st.dataframe(
        pd.DataFrame(
            [data_display]
        ),
        use_container_width=True
    )


    # ------------------------------------------
    # Analyze Selected Record
    # ------------------------------------------

    if st.button(
        "🔎 Analyze Selected Record",
        use_container_width=True
    ):

        features = [
            "Source Port",
            "Destination Port",
            "NAT Source Port",
            "NAT Destination Port",
            "Bytes",
            "Bytes Sent",
            "Bytes Received",
            "Packets",
            "Elapsed Time (sec)",
            "pkts_sent",
            "pkts_received"
        ]


        traffic_data = pd.DataFrame(
            [[
                selected_record[feature]
                for feature in features
            ]],
            columns=features
        )


        # --------------------------------------
        # Prediction
        # --------------------------------------

        prediction = model.predict(
            traffic_data
        )

        predicted_action = (
            label_encoder.inverse_transform(
                prediction
            )[0]
        )


        # --------------------------------------
        # Confidence
        # --------------------------------------

        probabilities = model.predict_proba(
            traffic_data
        )[0]

        confidence = (
            max(probabilities) * 100
        )


        # --------------------------------------
        # AI Prediction
        # --------------------------------------

        st.subheader(
            "🤖 AI Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                "Predicted Action: "
                f"{predicted_action.upper()}"
            )

        with col2:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


        # --------------------------------------
        # Prediction Comparison
        # --------------------------------------

        st.subheader(
            "📋 Prediction Comparison"
        )


        if predicted_action == actual_action:

            st.success(
                f"""
✓ Correct Prediction

Actual: {actual_action.upper()}

Predicted: {predicted_action.upper()}
"""
            )

        else:

            st.warning(
                f"""
Prediction differs from dataset label.

Actual: {actual_action.upper()}

Predicted: {predicted_action.upper()}
"""
            )


        # --------------------------------------
        # Firewall Recommendation
        # --------------------------------------

        if predicted_action == "allow":

            recommendation = "ALLOW"

        elif predicted_action == "deny":

            recommendation = "DENY"

        elif predicted_action == "drop":

            recommendation = "DROP"

        elif predicted_action == "reset-both":

            recommendation = "RESET CONNECTION"

        else:

            recommendation = "UNKNOWN"


        # --------------------------------------
        # Recommended Rule
        # --------------------------------------

        st.subheader(
            "🛡️ Recommended Firewall Rule"
        )

        st.info(
            f"""
**Action:** {recommendation}

**Source Port:** {selected_record['Source Port']}

**Destination Port:** {selected_record['Destination Port']}
"""
        )