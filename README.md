# AI-Teaching-Assistant
Developed a RAG-based AI teaching assistant that enables semantic search over lecture videos. The system processes user queries, retrieves the most relevant video segments using embeddings, and provides the exact video number, timestamp, and frame where the concept is discussed, allowing students to quickly navigate to the precise explanation.

# How to use this RAG AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM

## Using Precomputed Embeddings

This project includes **precomputed embeddings stored in `embeddings.joblib`**.  
The system loads these embeddings directly during query processing to perform semantic similarity search, avoiding the need to regenerate embeddings each time.
LINK : https://drive.google.com/file/d/16lO67HTDNp8a67Hjwygl5l2fzrWHarDT/view?usp=sharing
Embeddings only need to be regenerated if new videos or transcripts are added.

To regenerate embeddings (optional):

```bash
python embeddings.py



