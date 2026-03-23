import whisper

import json

import os

model = whisper.load_model("small")
audios = os.listdir("audios")
for audio in audios:
    print(audio)
    result = model.transcribe(audio = f"audios/{audio}",
                              language = "hi",
                              task = "translate",
                              word_timestamps = False
                                )
    
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "title": audio,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    print(chunks)

    chunks_with_metadata = {
        "chunks": chunks,             ##chunks of text with start and end time
        "text": result["text"]        ##saare text combined
        }


    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f, indent=4)