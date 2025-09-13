# 🩺 Healthcare Chatbot (Symptom-based Disease Prediction)


## 📌 Overview

This project is a healthcare chatbot that predicts possible diseases based on user-reported symptoms.
It uses a Decision Tree Classifier trained on a medical dataset with 132 symptoms and 41 diseases.
The chatbot interacts with the user, stores their symptoms, and predicts the disease when the user types done.
It also provides disease description and precautionary measures.


## 📂 Project Structure
```
CHATBOT/
│
├── Data/                        # Dataset folder
│   ├── Training.csv             # Training dataset (132 symptoms, 41 diseases)
│   ├── Testing.csv              # Testing dataset
│   ├── symptom_Description.csv  # Descriptions of diseases
│   ├── symptom_precaution.csv   # Precautions for each disease
│
├── chatbot.py                   # Main chatbot application
```


## ⚙️ Installation

1. Clone the repository or download the files.
```
git clone https://github.com/yourusername/healthcare-chatbot.git
cd healthcare-chatbot
```

3. Create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For Linux/Mac
```

4. Install dependencies:
```
pip install -r requirements.txt
```


## 🚀 Usage

1. Run the chatbot:
```
python chatbot.py
```

2. 💬 Example session:
```
You: itching
Bot: Added symptom 'itching'. Any other symptoms?

You: skin_rash
Bot: Added symptom 'skin_rash'. Any other symptoms?

You: nodal_skin_eruptions
Bot: Added symptom 'nodal_skin_eruptions'. Any other symptoms?

You: done
Bot: I think you may have Fungal infection.
Description: Fungal infections are caused by fungi affecting skin or scalp.
Precautions:
1) Bath twice
2) Keep area dry
3) Use clean clothes
4) Consult doctor
```


## 📊 Dataset Info

1. Training.csv

Rows: Thousands of records
Columns: 132 symptoms + 1 Disease label

2. Testing.csv

Similar format, used for evaluation

3. symptom_Description.csv

41 rows (Disease → Description)

4. symptom_precaution.csv

41 rows (Disease → 4 precautions)


## 🧠 How it Works

1. User enters symptoms one by one.

2. Symptoms are stored in a list.

3. When user types done, the chatbot:

-> Creates a feature vector (binary matrix for symptoms).
-> Runs it through the trained Decision Tree model.
-> Predicts the most likely disease.
-> Fetches description and precautions from CSV.

4. Displays results to the user.


## 🧪 Example Conversations
Example 1: Migraine
```
You: headache
Bot: Added symptom 'headache'. Any other symptoms?

You: nausea
Bot: Added symptom 'nausea'. Any other symptoms?

You: dizziness
Bot: Added symptom 'dizziness'. Any other symptoms?

You: blurred_and_distorted_vision
Bot: Added symptom 'blurred_and_distorted_vision'. Any other symptoms?

You: done
Bot: I think you may have Migraine.
Description: A headache of varying intensity, often accompanied by nausea and sensitivity to light and sound.
Precautions:
1) Medications
2) Avoid triggers
3) Reduce stress
4) Rest in dark room
```

Example 2: Diabetes
```
You: fatigue
Bot: Added symptom 'fatigue'. Any other symptoms?

You: weight_loss
Bot: Added symptom 'weight_loss'. Any other symptoms?

You: excessive_hunger
Bot: Added symptom 'excessive_hunger'. Any other symptoms?

You: polyuria
Bot: Added symptom 'polyuria'. Any other symptoms?

You: done
Bot: I think you may have Diabetes.
Description: A group of diseases that result in too much sugar in the blood (high blood glucose).
Precautions:
1) Maintain diet
2) Exercise
3) Monitor blood sugar
4) Take insulin
```


## ⚡ Execution Commands

```
cd C:\Users\DELL\Desktop\CHATBOT
python chatbot.py
```

```
cd C:\Users\DELL\Desktop\CHATBOT
python demo_conversations.py
```


## 📜 License
This project is licensed under the MIT License
