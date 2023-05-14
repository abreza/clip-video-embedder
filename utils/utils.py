import os
import json
import pickle


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



def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]
