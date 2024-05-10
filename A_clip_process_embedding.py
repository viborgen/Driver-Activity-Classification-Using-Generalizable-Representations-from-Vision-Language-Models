import numpy as np
import pandas as pd
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
import os
import cv2

def get_class(df, frame_number):
    # Find the row where the frame number falls between the start and end frames
    row = df[(df['Start Frame'] <= frame_number) & (df['End Frame'] >= frame_number)]
    
    # If the frame number falls within a labeled interval, return the class number
    if not row.empty:
        class_str = row['Label (Primary)'].values[0]  # This should be a string like 'Class 8'
        class_number = int(class_str.split(' ')[1])+1  # Split the string and convert the second part to an integer. Adding 1 to leave 0 for background class
        return class_number
    
    # If the frame number does not fall within any labeled interval, return 0 for straight forward driving
    return 0 

# Generate CLIP embeddings for all images, handling damaged images if any
def generate_embeddings(video_id, batch_size, user_id, last_value, df):  
    all_embeddings = []
    class_values = [] 
    progress_bar = tqdm(total=int(last_value/30), desc="Generating CLIP embeddings")
    # Open the video file
    cap = cv2.VideoCapture(rootfolder + "/" + user_id + "/" + video_id)
    frame_num = 0
    ret = True
    while ret: 
        batch_images = [] 
        for ii in range(batch_size):
            ret, frame = cap.read()
            frame_num += 1
            if not ret:
                break
            else:
                batch_images += [frame]
                class_values += [get_class(df, frame_num)]
                #frame_num += 1
        if not batch_images:  # Check if the list is empty
            print(f"No frames could be read from video {video}. Skipping...")
            continue  # Skip the rest of the loop and continue with the next video
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        all_embeddings.extend(outputs.cpu().numpy())
        progress_bar.update(len(batch_images))
    progress_bar.close()
    return all_embeddings, class_values

clip_model = "openai/clip-vit-large-patch14-336"
batch_size = 256
device = "cuda"

# Load CLIP model and processor
model = CLIPModel.from_pretrained(clip_model).to(device)
processor = CLIPProcessor.from_pretrained(clip_model)
baseFolder = ''
for suffix in ["A1_1","A1_2","A1_3","A1_4","A1_5","A1_6","A1_7"]:
    rootfolder = baseFolder + suffix
    for user_id in os.listdir(rootfolder): 
        print(f'{user_id} is running')
        for video in os.listdir(rootfolder + "/" + user_id):
            print(f'{video} is running')
            csvPath = "data/labels&instructions/A1/A1/user_id_" + video[video.index("_NoAudio")-5:video.index("_NoAudio")] + ".csv"
            # Load the CSV file
            df = pd.read_csv(csvPath)

            # Convert the start and end times to frame numbers
            df['Start Frame'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.hour * 3600 * 30 + pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.minute * 60 * 30 + pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.second * 30
            df['End Frame'] = pd.to_datetime(df['End Time'], format='%H:%M:%S').dt.hour * 3600 * 30 + pd.to_datetime(df['End Time'], format='%H:%M:%S').dt.minute * 60 * 30 + pd.to_datetime(df['End Time'], format='%H:%M:%S').dt.second * 30
            last_value = df.iloc[-1]['End Frame']

            embeddings, labels = generate_embeddings(video, batch_size, user_id, last_value, df)
            # Save embeddings
            save_dir = f'/home/cvrr/Desktop/CVPRchallenge/CVPRchallenge/clip_embeddings{suffix}'
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Save embeddings
            np.savez_compressed(f'{save_dir}/embeddingsandlabels30_{video[:-4]}.npz', embeddings=embeddings, labels=labels)
