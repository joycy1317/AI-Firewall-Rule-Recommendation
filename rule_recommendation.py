import pandas as pd
import joblib


# ==========================================
# 1. Load Dataset
# ==========================================

DATASET_PATH = "results/processed_firewall_logs.csv"
MODEL_PATH = "models/firewall_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


df = pd.read_csv(DATASET_PATH)

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ==========================================
# 2. Select Features
# ==========================================

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


# ==========================================
# 3. Function to Predict Firewall Action
# ==========================================

def predict_action(traffic_data):

    input_data = pd.DataFrame(
        [traffic_data],
        columns=features
    )

    prediction = model.predict(input_data)

    predicted_action = label_encoder.inverse_transform(
        prediction
    )[0]

    return predicted_action


# ==========================================
# 4. Generate Firewall Rule
# ==========================================

def generate_rule(traffic_data, predicted_action):

    source_port = traffic_data["Source Port"]
    destination_port = traffic_data["Destination Port"]

    if predicted_action == "allow":

        recommendation = "ALLOW"

        reason = (
            "The traffic pattern was classified as allowed "
            "by the machine learning model."
        )

    elif predicted_action == "deny":

        recommendation = "DENY"

        reason = (
            "The traffic pattern was classified as denied "
            "by the machine learning model."
        )

    elif predicted_action == "drop":

        recommendation = "DROP"

        reason = (
            "The traffic pattern was classified as drop "
            "by the machine learning model."
        )

    elif predicted_action == "reset-both":

        recommendation = "RESET CONNECTION"

        reason = (
            "The traffic pattern was classified as reset-both "
            "by the machine learning model."
        )

    else:

        recommendation = "UNKNOWN"

        reason = "The predicted action is not recognized."


    return {
        "action": recommendation,
        "source_port": source_port,
        "destination_port": destination_port,
        "reason": reason
    }


# ==========================================
# 5. Test With One Existing Traffic Record
# ==========================================

sample = df.drop("Action", axis=1).iloc[0].to_dict()

predicted_action = predict_action(sample)

rule = generate_rule(
    sample,
    predicted_action
)


# ==========================================
# 6. Display Recommendation
# ==========================================

print("\n==========================================")
print("     AI FIREWALL RULE RECOMMENDATION")
print("==========================================")

print("\nTraffic Information:")
print("Source Port       :", sample["Source Port"])
print("Destination Port  :", sample["Destination Port"])
print("Bytes             :", sample["Bytes"])
print("Packets           :", sample["Packets"])
print("Elapsed Time      :", sample["Elapsed Time (sec)"])

print("\nML Prediction:")
print(predicted_action.upper())

print("\nRecommended Firewall Rule:")
print("------------------------------------------")
print("Action             :", rule["action"])
print("Source Port        :", rule["source_port"])
print("Destination Port   :", rule["destination_port"])

print("\nReason:")
print(rule["reason"])

print("------------------------------------------")