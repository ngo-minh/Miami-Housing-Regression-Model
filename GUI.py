import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from joblib import load

# load both predictive models
def load_models():
    models = {}
    models['Decision Tree'] = load("C:/Users/mnm4m/project/best_model.joblib")
    models['Random Forest'] = load("C:/Users/mnm4m/project/best_model2.joblib")
    return models

# initialize the models dictionary
models = load_models()
# set the initial model
current_model = models['Random Forest']

# setup the main application window
root = tk.Tk()
root.title("Miami Predictive Housing Model")
root.configure(bg='white')

# ensure logo_photo is globally accessible to prevent garbage collection
global logo_photo

# load and display the application logo
logo_path = "C:/Users/mnm4m/project/logo.jpg"
logo_image = Image.open(logo_path).resize((100, 100))
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(root, image=logo_photo, bg='white')
logo_label.pack(side="top", anchor="nw")

# load and prepare the reset icon
reset_icon_path = "reset.png"
reset_icon_image = Image.open(reset_icon_path).resize((30, 30))
reset_icon_photo = ImageTk.PhotoImage(reset_icon_image)

# configure ttk styles for consistency in the app
style = ttk.Style()
style.configure("TLabel", background='white', font=('Calibri', 10, 'bold'))
style.configure("TLabelframe", background='white', font=('Calibri', 10, 'bold'))
style.configure("TLabelframe.Label", background='white', font=('Calibri', 10, 'bold'))
style.configure("TEntry", fieldbackground='white', background='white', font=('Calibri', 10, 'bold'))
style.configure("TCheckbutton", background='white', font=('Calibri', 10, 'bold'))

# reset form fields to initial state
def reset_form():
    for entry in entries:
        entry.delete(0, tk.END)
    result_label.config(text="Enter your inputs")

# use the model to predict house prices
def predict_house_price(input_data):
    input_scaled = pre_input(input_data)
    return current_model.predict(input_scaled)[0]

# preprocess the input data
def pre_input(input_data):
    full_input = np.full((1, 11), np.nan)
    active_indices = [i for i, chk in enumerate(checkbox_states) if not chk.get()]
    for index, value in zip(active_indices, input_data):
        full_input[0, index] = value
    return full_input

# display the prediction or error messages
def display_info():
    if all(not entry.get().strip() for entry, chk in zip(entries, checkbox_states) if not chk.get()):
        result_label.config(text="Please input house details or adjust 'Omit' selections.")
        return

    try:
        values = [float(entry.get()) if not chk.get() else np.nan for entry, chk in zip(entries, checkbox_states)]
        if all(np.isnan(v) for v in values):
            raise ValueError("No valid inputs selected.")
        predicted_price = predict_house_price(values)
        result_label.config(text=f"Predicted Price: ${predicted_price:.2f}")
    except ValueError as e:
        result_label.config(text=str(e))

# model selection
def update_model(event):
    global current_model
    current_model = models[model_selector.get()]


# model selection setup
model_label = ttk.Label(root, text="Select Model:", background='white', font=('Calibri', 10, 'bold'))
model_label.pack(pady=(5, 0))

model_selector = ttk.Combobox(root, values=list(models.keys()), state="readonly")
model_selector.current(1)  # default to Random Forest
model_selector.pack(pady=10)
model_selector.bind("<<ComboboxSelected>>", update_model)

# create and organize input fields within a label frame
input_frame = ttk.LabelFrame(root, text="Input Fields", style="TLabelframe")
input_frame.pack(padx=10, pady=(10, 0), fill='x', expand=True)

omit_label = ttk.Label(input_frame, text="Omit", style="TLabel")
omit_label.grid(row=0, column=2, pady=5)

reset_button = tk.Button(root, image=reset_icon_photo, command=reset_form, bg='white', bd=0)
reset_button.image = reset_icon_photo 
reset_button.pack(side='right', anchor='ne', padx=(0, 14), pady=(0, 10))

# define fields and checkboxes for user input
entries = []
checkbox_states = []
fields = [
    ("LND_SQFOOT", "Land Area (sq ft)"), ("TOT_LVG_AREA", "Floor Area (sq ft)"),
    ("SPEC_FEAT_VAL", "Value of Special Features ($)"), ("RAIL_DIST", "Distance to Nearest Rail Line (ft)"),
    ("OCEAN_DIST", "Distance to the Ocean (ft)"), ("WATER_DIST", "Distance to Nearest Body of Water (ft)"),
    ("CNTR_DIST", "Distance to Miami CBD (ft)"), ("SUBCNTR_DI", "Distance to Nearest Subcenter (ft)"),
    ("HWY_DIST", "Distance to Nearest Highway (ft)"), ("age", "Age of the Structure (years)"),
    ("structure_quality", "Structure Quality (1-5 scale)")
]

# dynamically create input fields and checkboxes
for i, (field, label) in enumerate(fields):
    ttk.Label(input_frame, text=f"{label}:", style="TLabel").grid(row=i+1, column=0, pady=5)
    entry = ttk.Entry(input_frame, style="TEntry")
    entry.grid(row=i+1, column=1, pady=5)
    entries.append(entry)
    chk_var = tk.IntVar(value=0)
    chk = ttk.Checkbutton(input_frame, style="TCheckbutton", variable=chk_var)
    chk.grid(row=i+1, column=2)
    checkbox_states.append(chk_var)

spacer_frame = tk.Frame(root, bg='white', width=35)
spacer_frame.pack(side='left', fill='y')
spacer_frame.pack_propagate(False)

submit_button = tk.Button(root, text="Submit", command=display_info, bg='white', font=('Calibri', 10, 'bold'))
submit_button.pack(pady=5)

result_label = tk.Label(root, text="Enter your inputs", bg='white', font=('Calibri', 10, 'bold'))
result_label.pack(pady=5)

root.mainloop()
