import tkinter as tk
from tkinter import scrolledtext
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
symptoms_list = X.columns.tolist()
collected_symptoms = []

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

def on_send():
    user = entry.get().strip()
    if not user:
        return
    
    chat_box.configure(state='normal')
    chat_box.insert(tk.END, "You: " + user + "\n")
    
    global collected_symptoms

    if user.lower() in symptoms_list:
        # Add symptom if it's valid
        collected_symptoms.append(user.lower())
        chat_box.insert(tk.END, f"Bot: Added symptom '{user}'. Any other symptoms?\n\n")

    elif user.lower() == "done":
        # Predict the disease
        input_data = [0] * len(symptoms_list)
        for s in collected_symptoms:
            if s in symptoms_list:
                input_data[symptoms_list.index(s)] = 1
        disease = model.predict([input_data])[0]

        # Get details
        description = get_description(disease)
        precautions = get_precautions(disease)

        chat_box.insert(tk.END, f"Bot: I think you may have **{disease}**.\n")
        chat_box.insert(tk.END, f"Description: {description}\n")
        chat_box.insert(tk.END, "Precautions:\n")
        for i, p in enumerate(precautions, 1):
            chat_box.insert(tk.END, f"{i}) {p}\n")
        chat_box.insert(tk.END, "\n")

        # Reset for next conversation
        collected_symptoms = []

    else:
        chat_box.insert(
            tk.END,
            "Bot: Sorry, I don’t recognize that symptom. "
            "Try another or type 'done'.\n\n"
        )
    
    chat_box.configure(state='disabled')
    entry.delete(0, tk.END)
    chat_box.see(tk.END)

# --- Tkinter UI ---
root = tk.Tk()
root.title("AI Health Assistant (Interactive)")
root.geometry("600x500")

chat_box = scrolledtext.ScrolledText(root, state='disabled', wrap='word')
chat_box.pack(padx=10, pady=10, fill='both', expand=True)

entry_frame = tk.Frame(root)
entry_frame.pack(padx=10, pady=(0, 10), fill='x')

entry = tk.Entry(entry_frame)
entry.pack(side='left', fill='x', expand=True, padx=(0, 6))
entry.bind("<Return>", lambda event: on_send())

send_btn = tk.Button(entry_frame, text="Send", command=on_send)
send_btn.pack(side='right')

# Initial message
chat_box.configure(state='normal')
chat_box.insert(tk.END, "Bot: Hello! Please tell me your symptoms one by one.\n")
chat_box.insert(tk.END, "When you are done, type 'done'.\n\n")
chat_box.configure(state='disabled')

root.mainloop()
