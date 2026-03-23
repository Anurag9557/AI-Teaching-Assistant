import requests
import os
import json

def create_embeddings(text):
    r = requests.post("http://localhost:11434/api/embeddings" ,
                      json = {
                          "model": "bge-m3",
                          "prompt": "text"
                      }
                      ) 
    embeddings = r.json()['embedding']
    return embeddings
my_dicts = []
jsons = os.listdir("jsons")
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    for chunk in content['chunks']:
        my_dicts.append(chunk)
    break

print(my_dicts)