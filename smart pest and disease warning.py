# ===============================================
# ADVANCED SMART PEST & DISEASE EARLY WARNING SYSTEM
# AI-Driven | Explainable | Visual Dashboard
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------
# 1. DATA GENERATION (SMART AGRI DATA)
# -----------------------------------------------
np.random.seed(42)

data = pd.DataFrame({
    "Temperature (°C)": np.random.randint(20, 38, 300),
    "Humidity (%)": np.random.randint(50, 95, 300),
    "Rainfall (mm)": np.random.randint(0, 90, 300)
})

# Risk labeling (realistic logic)
conditions = [
    (data["Humidity (%)"] > 80) & (data["Temperature (°C)"] > 28),
    (data["Humidity (%)"] > 65) & (data["Rainfall (mm)"] > 40)
]
choices = [2, 1]  # High, Medium
data["Risk"] = np.select(conditions, choices, default=0)

# -----------------------------------------------
# 2. MODEL TRAINING (ADVANCED RANDOM FOREST)
# -----------------------------------------------
X = data[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]]
y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# Model Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# -----------------------------------------------
# 3. PREDICTION FUNCTION
# -----------------------------------------------
def predict_risk(temp, humidity, rainfall):
    input_data = pd.DataFrame(
        [[temp, humidity, rainfall]],
        columns=X.columns
    )
    probabilities = model.predict_proba(input_data)[0]
    risk_index = np.argmax(probabilities)
    risk_levels = ["Low", "Medium", "High"]
    return risk_levels[risk_index], probabilities

# -----------------------------------------------
# 4. SUSTAINABLE ADVISORY SYSTEM
# -----------------------------------------------
def sustainable_advisory(risk):
    advice = {
        "High": "High risk detected. Use biological controls, neem-based sprays, improve airflow.",
        "Medium": "Moderate risk. Increase field monitoring and use trap crops.",
        "Low": "Low risk. Maintain current sustainable practices."
    }
    return advice[risk]

# -----------------------------------------------
# 5. ADVANCED VISUAL DASHBOARD
# -----------------------------------------------
def visualize_dashboard(temp, humidity, rainfall, risk, probs):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Environmental Conditions
    axs[0, 0].bar(
        ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)"],
        [temp, humidity, rainfall],
        color=["orange", "blue", "green"]
    )
    axs[0, 0].set_title("Environmental Conditions")

    # Risk Probability
    axs[0, 1].bar(
        ["Low", "Medium", "High"],
        probs,
        color=["green", "gold", "red"]
    )
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_title(" Risk Probability")

    # Feature Importance (Explainable )
    sns.barplot(
        x=model.feature_importances_,
        y=X.columns,
        ax=axs[1, 0],
        palette="viridis"
    )
    axs[1, 0].set_title("Feature Importance (Explainable )")

    # Risk Indicator Gauge (Simplified)
    risk_score = {"Low": 0.3, "Medium": 0.6, "High": 0.9}[risk]
    axs[1, 1].barh(["Risk Level"], [risk_score],
                   color="red" if risk == "High" else "orange")
    axs[1, 1].set_xlim(0, 1)
    axs[1, 1].set_title("Final Risk Indicator")

    fig.suptitle(
        f"FINAL PREDICTION: {risk.upper()} RISK\nModel Accuracy: {accuracy:.2%}",
        fontsize=18,
        color="red" if risk == "High" else "orange"
    )

    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------
print("\nADVANCED SMART PEST & DISEASE EARLY WARNING SYSTEM\n")

temp = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
rainfall = float(input("Enter Rainfall (mm): "))

risk, probabilities = predict_risk(temp, humidity, rainfall)

print("\n--- RESULTS ---")
print("Predicted Risk Level:", risk)
print("Model Accuracy:", f"{accuracy:.2%}")
print("SUSTAINABLE ADVISORY:")
print(sustainable_advisory(risk))

visualize_dashboard(temp, humidity, rainfall, risk, probabilities)