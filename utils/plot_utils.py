import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_text_image_pallet(model, processor, images, sentences,mode=None):

    processed_images = processor(images =images)['pixel_values']

    image_input = torch.squeeze(torch.tensor(np.stack(processed_images)), dim=1).to(model.device)
    text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        image_features = model.get_image_features(image_input).float()
        text_features = model.get_text_features(**text_inputs).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    if mode== 'max':
      similarity = np.max(similarity, axis=0)
      similarity = similarity.reshape((1, similarity.shape[0]))

    plt.figure(figsize=(10, 10))
    plt.imshow(similarity, vmin=np.min(similarity), vmax=np.max(similarity))

    if similarity.shape[1] > 15:
        plt.yticks([])
    else:
        plt.yticks(range(similarity.shape[0]), sentences, fontsize=18)
    plt.xticks([])

    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    if len(images)<30:
      for x in range(similarity.shape[1]):
          for y in range(similarity.shape[0]):
              plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=10)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, similarity.shape[1] - 0.5]);
    plt.ylim([similarity.shape[0] - 0.5, -1.6]);

def plot_image_image_pallet(model, processor, images):  
   
    processed_images = processor(images=images)['pixel_values']

    image_input = torch.squeeze(torch.tensor(np.stack(processed_images)), dim=1).to(model.device)

    with torch.no_grad():
        image_features = model.get_image_features(image_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity = image_features.cpu().numpy() @ image_features.cpu().numpy().T

    plt.figure(figsize=(10, 10))
    plt.imshow(similarity, vmin=np.min(similarity), vmax=np.max(similarity))

    for i, image in enumerate(images):
        plt.imshow(image, extent=(-1.6, -0.6, i - 0.5, i + 0.5), origin="lower")
        
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    if len(images)<30:
      for x in range(similarity.shape[1]):
          for y in range(similarity.shape[0]):
              plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=10)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.yticks([])
    plt.xticks([])

    plt.xlim([-1.6, similarity.shape[1] - 0.5]);
    plt.ylim([similarity.shape[0] - 0.5, -1.6]);

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

def plot_student_teacher_scores(teacher_scores, student_scores,framerate):
              
    fig , ax = plt.subplots(1, 1, figsize=(20, 4))

    ax.plot( np.arange(len(student_scores))/ framerate, student_scores, label='Student Saliency Score')
    ax.plot( np.arange(len(teacher_scores))/ framerate, student_scores, label='Teacher Saliency Score')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("saliency score")
    ax.set_xticks(np.arange(0, round(len(student_scores)/ framerate) + 1, 5))
    ax.legend()

    plt.show()