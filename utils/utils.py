import json
import numpy as np
import matplotlib.pyplot as plt

def trim_sentences(sentences, max_seq_len=75):
    trimmed_sentences = []
    for sentence in sentences:
        tokens = sentence.strip().split()[:max_seq_len]
        trimmed_sentence = " ".join(tokens)
        trimmed_sentences.append(trimmed_sentence)
    return trimmed_sentences


def write_results_to_json(key_names, description, similarity_scores, timestamps, output_file, indent=1):
    descs = []
    for i, key_name in enumerate(key_names):
        desc = {"name": key_name, "sentence": description[i]}
        if key_name.startswith("Description"):
            idx = int(key_name.split(" ")[1]) - 1
            desc["timestamp"] = timestamps[idx]
        descs.append(desc)

    results = {}
    for i, key_name in enumerate(key_names):
        results[key_name] = similarity_scores[i].tolist()

    data = {"descs": descs, "results": results}
    with open(output_file, "w") as f:
        json.dump(data, f, indent=indent)


def load_dataset(dataset_name):
  if dataset_name == "ActivityNet":
    with open('datasets/ActivityNet/train.json') as f:
      loaded_file = json.load(f)

      data = {}
      for id in loaded_file:
        instance = {}
        instance['start'] = 0.0
        instance['end'] = loaded_file[id]['duration']
        instance['descriptions'] = loaded_file[id]['sentences']

        data[id] = instance

  elif dataset_name == "MSVD":
    with open('datasets/MSVD/data.json') as f:
      loaded_file = json.load(f)

      data = {}
      for video_info in loaded_file:
        instance = {}
        id = video_info['link']
        instance['start'] = video_info['start']
        instance['end'] = video_info['end']
        instance['descriptions'] = video_info['description']

        data[id] = instance

  return data