# chatbot.py (Streamlit version)
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --- Load datasets ---
train = pd.read_csv("Data/Training.csv")    # main training data
test = pd.read_csv("Data/Testing.csv")      # optional test data
desc = pd.read_csv("Data/symptom_Description.csv")
prec = pd.read_csv("Data/symptom_precaution.csv")

# Features (symptoms) and Labels (diseases)
X = train.drop("prognosis", axis=1)
y = train["prognosis"]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# --- Chatbot Logic ---
symptoms_list = [s.lower() for s in X.columns]  # all symptoms lowercase

def get_precautions(disease):
    row = prec[prec["Disease"] == disease]
    if not row.empty:
        return [row[f"Precaution_{i}"].values[0] for i in range(1, 5)]
    return ["No precautions available."]

def get_description(disease):
    row = desc[desc["Disease"] == disease]
    if not row.empty:
        return row["Description"].values[0]
    return "No description available."

# --- Streamlit UI ---
st.set_page_config(page_title="AI Health Assistant", layout="centered")
st.title("🩺 AI Health Assistant")

# Initialize session state
if "collected_symptoms" not in st.session_state:
    st.session_state.collected_symptoms = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(chat)

# User input
user_input = st.text_input("Type your symptom (or type 'done'):")

if st.button("Send") and user_input:
    user = user_input.strip().lower()
    chat_output = ""

    # Save user's message
    chat_output += f"**You:** {user_input}\n\n"

    if user in symptoms_list:
        st.session_state.collected_symptoms.append(user)
        chat_output += f"**Bot:** Added symptom '{user_input}'. Any other symptoms?\n\n"

    elif user == "done":
        # Predict the disease
        input_data = [0] * len(symptoms_list)
        for s in st.session_state.collected_symptoms:
            if s in symptoms_list:
                input_data[symptoms_list.index(s)] = 1
        disease = model.predict([input_data])[0]

        # Get details
        description = get_description(disease)
        precautions = get_precautions(disease)

        chat_output += f"**Bot:** I think you may have **{disease}**.\n\n"
        chat_output += f"**Description:** {description}\n\n"
        chat_output += "**Precautions:**\n"
        for i, p in enumerate(precautions, 1):
            chat_output += f"{i}) {p}\n"

        # Reset for next conversation
        st.session_state.collected_symptoms = []

    else:
        chat_output += (
            "**Bot:** Sorry, I don’t recognize that symptom. "
            "Try another or type 'done'.\n\n"
        )

    # Save chat history
    st.session_state.chat_history.append(chat_output)

    # Clear input box
    st.experimental_rerun()
