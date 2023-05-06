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

def plot_clip_similarities(outputs, sentences, random_sentences=None,
                       timestamps=[], framerate=1,
                       force_separate_subplots = False,
                       show_random_plots = False,
                       show_concate_descriptions_plot=False,
                       show_average_plot = False, 
                       show_max_plot = False,
                       show_each_plot=False):
           
    concat_output = outputs[0].tolist()
    descriptions_output = np.array([output.tolist() for output in outputs[1:]])

    t = np.arange(outputs.shape[1])

    _temp = len(t)/framerate
    interval_length = 1 if _temp<25 else 2 if _temp < 50 else 5 if _temp < 100 else 10 if _temp <260 else 10 if _temp<360 else 20
    plot_w = 12 if _temp<150 else 18 if _temp<350 else 22


    if force_separate_subplots:

        number_of_subplots = len(sentences)+len(random_sentences) if random_sentences and show_random_plots else len(sentences)

        fig , axs = plt.subplots(number_of_subplots, 1, figsize=(plot_w, number_of_subplots*2))

        for i,rasentence in enumerate(sentences):
          axs[i].plot(t/ framerate, descriptions_output[i])
          axs[i].set_title(sentences[i])
          axs[i].set_xlabel("Time (s)")
          axs[i].set_ylabel("Similarity")
          axs[i].axvline(x=timestamps[i][0] , color='r', linestyle='--')
          axs[i].axvline(x=timestamps[i][1] , color='r', linestyle='--')
          axs[i].set_xticks(np.arange(0, round(len(t)/ framerate) + 1, interval_length))


        if random_sentences and show_random_plots:        
          for i, random_sent in enumerate(random_sentences):
            index = i + len(sentences)
            axs[index].plot(t/ framerate, descriptions_output[index], color='orange')
            axs[index].set_title('Random Sentence: '+random_sent)
            axs[index].set_xlabel("Time (s)")
            axs[index].set_ylabel("similarity")
            axs[index].set_xticks(np.arange(0, round(len(t)/ framerate) + 1, interval_length))


    else:
        
        fig , ax = plt.subplots(1, 1, figsize=(plot_w, 4))

        if show_concate_descriptions_plot:
            ax.plot(t/ framerate, concat_output, label='Concatenated Descriptions')

        if show_average_plot:
            mean_of_lists = np.mean(descriptions_output[:len(sentences)], axis=0)
            ax.plot(t/ framerate, mean_of_lists, label=f'Average of {len(sentences)} Plots')
        
        if show_max_plot:
            max_of_lists = np.max(descriptions_output[:len(sentences)], axis=0)
            ax.plot(t/ framerate, max_of_lists, label=f'Max of {len(sentences)} Plots')

        if show_each_plot:
            for i in range(len(sentences)):
                ax.plot(t/ framerate, descriptions_output[i], label=f'Description {i+1}')

        if random_sentences and show_random_plots:
          if len(random_sentences) == 1:
            ax.plot(t/ framerate, descriptions_output[len(sentences)], label=f'Random Description')
          
          else:  
            for i in range(len(random_sentences)):
              index = len(sentences)+i
              ax.plot(t/ framerate, descriptions_output[index], label=f'Random Description {i+1}')

        ax.set_title("CLIP Text-Frame Cosine Similarity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("text-frame similarity")
        ax.set_xticks(np.arange(0, round(len(t)/ framerate) + 1, interval_length))
        ax.legend()

    fig.subplots_adjust(hspace=1)
    plt.show()