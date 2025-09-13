import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --- Load datasets ---
train = pd.read_csv("Data/Training.csv")
desc = pd.read_csv("Data/symptom_Description.csv")
prec = pd.read_csv("Data/symptom_precaution.csv")

# Features & labels
X = train.drop("prognosis", axis=1)
y = train["prognosis"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Symptom list
symptoms_list = X.columns.tolist()

# --- Helper functions ---
def get_description(disease):
    row = desc[desc["Disease"] == disease]
    if not row.empty:
        return row["Description"].values[0]
    return "No description available."

def get_precautions(disease):
    row = prec[prec["Disease"] == disease]
    if not row.empty:
        return [row[f"Precaution_{i}"].values[0] for i in range(1, 5)]
    return ["No precautions available."]

def predict_disease(symptom_inputs):
    input_data = [0] * len(symptoms_list)
    for s in symptom_inputs:
        if s in symptoms_list:
            input_data[symptoms_list.index(s)] = 1
    disease = model.predict([input_data])[0]
    return disease

# --- 50 Sample conversations ---
conversations = [
    ["itching", "skin_rash", "nodal_skin_eruptions"],  # Fungal infection
    ["headache", "nausea", "dizziness", "blurred_and_distorted_vision"],  # Migraine
    ["yellowing_of_eyes", "dark_urine", "abdominal_pain", "vomiting"],  # Jaundice
    ["fatigue", "weight_loss", "restlessness", "mood_swings"],  # Diabetes
    ["cough", "high_fever", "breathlessness", "sweating"],  # Tuberculosis
    ["loss_of_appetite", "stomach_pain", "weight_loss", "abdominal_pain"],  # Peptic ulcer
    ["chills", "sweating", "headache", "vomiting"],  # Malaria
    ["joint_pain", "swelling_joints", "stiffness", "movement_stiffness"],  # Arthritis
    ["vomiting", "abdominal_pain", "diarrhea", "indigestion"],  # Gastroenteritis
    ["yellowing_of_eyes", "vomiting", "dark_urine", "fatigue"],  # Hepatitis
    ["sore_throat", "cough", "fever", "congestion"],  # Common Cold
    ["shortness_of_breath", "chest_pain", "fatigue", "sweating"],  # Heart Disease
    ["anxiety", "restlessness", "sleeplessness", "irritability"],  # Anxiety Disorder
    ["weight_gain", "fatigue", "cold_hands_and_feets", "swelling_of_body"],  # Hypothyroidism
    ["weight_loss", "increased_appetite", "sweating", "restlessness"],  # Hyperthyroidism
    ["bloody_stool", "abdominal_pain", "fatigue", "weight_loss"],  # Ulcerative Colitis
    ["nausea", "vomiting", "stomach_pain", "loss_of_appetite"],  # Food Poisoning
    ["rash", "fever", "body_pain", "joint_pain"],  # Dengue
    ["chest_pain", "sweating", "shortness_of_breath", "nausea"],  # Heart Attack
    ["fever", "chills", "sweating", "muscle_pain"],  # Typhoid
    ["sneezing", "runny_nose", "watery_eyes", "itching"],  # Allergy
    ["back_pain", "weakness_in_limbs", "neck_pain", "dizziness"],  # Cervical spondylosis
    ["skin_peeling", "silver_like_dusting", "inflammatory_nails", "joint_pain"],  # Psoriasis
    ["sunken_eyes", "dehydration", "diarrhea", "fatigue"],  # Dehydration
    ["cough", "fever", "difficulty_breathing", "fatigue"],  # Pneumonia
    ["blood_in_sputum", "fatigue", "loss_of_appetite", "chest_pain"],  # Tuberculosis variant
    ["stomach_pain", "nausea", "vomiting", "loss_of_appetite"],  # Gastritis
    ["headache", "irritability", "anxiety", "depression"],  # Mental health case
    ["itching", "skin_rash", "dischromic_patches"],  # Dermatophytosis
    ["ulcers_on_tongue", "vomiting", "fatigue", "nausea"],  # Gastro issues
    ["burning_micturition", "bladder_discomfort", "continuous_feel_of_urine"],  # UTI
    ["abdominal_pain", "diarrhea", "indigestion", "nausea"],  # IBS
    ["patches_in_throat", "fever", "swelled_lymph_nodes"],  # Tonsillitis
    ["neck_pain", "dizziness", "headache", "back_pain"],  # Spondylitis
    ["vomiting", "yellowing_of_eyes", "fatigue", "nausea"],  # Hepatitis B
    ["joint_pain", "back_pain", "weakness_in_limbs"],  # Arthritis variant
    ["high_fever", "rash", "body_pain", "headache"],  # Viral Fever
    ["loss_of_balance", "unsteadiness", "headache"],  # Vertigo
    ["altered_sensorium", "headache", "nausea"],  # Neurological disorder
    ["weight_loss", "chills", "fatigue", "cough"],  # TB-like
    ["muscle_wasting", "weakness", "fatigue"],  # Muscular dystrophy
    ["abdominal_pain", "vomiting", "fatigue"],  # Gastro disease
    ["palpitations", "fatigue", "sweating"],  # Cardiac issue
    ["constipation", "stomach_pain", "gas", "acidity"],  # Digestive issue
    ["mood_swings", "restlessness", "weight_loss"],  # Hyperthyroid pattern
    ["sleeplessness", "fatigue", "anxiety"],  # Insomnia
    ["headache", "nausea", "sensitivity_to_light"],  # Migraine variant
    ["bloody_stool", "vomiting", "stomach_pain"],  # Gastro bleeding
    ["swelling_joints", "painful_walking", "stiffness"],  # Rheumatoid Arthritis
    ["fatigue", "chest_pain", "shortness_of_breath"],  # Heart Disease
]

# --- Run demo ---
for i, symptoms in enumerate(conversations, 1):
    print(f"\n--- Conversation {i} ---")
    for s in symptoms:
        print(f"You: {s}")
        print(f"Bot: Added symptom '{s}'. Any other symptoms?")
    print("You: done")
    
    disease = predict_disease(symptoms)
    description = get_description(disease)
    precautions = get_precautions(disease)

    print(f"Bot: I think you may have **{disease}**.")
    print(f"Description: {description}")
    print("Precautions:")
    for j, p in enumerate(precautions, 1):
        print(f"{j}) {p}")
