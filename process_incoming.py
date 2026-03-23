from logging import root
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1", #time consuming
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    #print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching Data Science course. Here are video subtitle chunks containing video title,start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
#print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

print("Response is saved to response.txt")

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk

# --- Sample response (replace this with your variable) ---

# Create main window
root = tk.Tk()
root.title("✨ Response Viewer ✨")

# Set window size and center it
window_width = 650
window_height = 550
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = int((screen_width / 2) - (window_width / 2))
y_position = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
root.configure(bg="#1e1e2f")  # Dark modern background

# Use a modern style
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 11), padding=6, relief="flat", background="#4CAF50")
style.map("TButton", background=[("active", "#45a049")])

# Title label
title_label = tk.Label(
    root,
    text="📜 Response",
    font=("Segoe UI", 16, "bold"),
    bg="#1e1e2f",
    fg="white"
)
title_label.pack(pady=(15, 5))

# Scrolled text area
text_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=75,
    height=20,
    font=("Consolas", 12),
    bg="#2d2d44",
    fg="#f5f5f5",
    insertbackground="white",  # Cursor color
    relief="flat",
    bd=8
)
text_area.pack(padx=15, pady=10, fill="both", expand=True)

# Insert response and lock editing
text_area.insert(tk.INSERT, response)
text_area.configure(state="disabled")

# Close button
close_btn = ttk.Button(root, text="Close", command=root.destroy)
close_btn.pack(pady=10)

# Run
root.mainloop()



