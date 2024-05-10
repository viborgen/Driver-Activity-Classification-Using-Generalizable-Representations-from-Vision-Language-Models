import os
import numpy as np
"""
Use this file by running it and specifying which of the 7 directories you want as the testing directory.
Example: python B_combine_npz.py 1 will utilize fold 1 as the test directory. If test directory 10 is specified all directories will be utilized for training.

"""



def gatherTrain(directories, saveFolder):
    # Initialize variables to store embeddings and labels for each view
    total_dashboard_embeddings = []
    total_rearview_embeddings = []
    total_side_embeddings = []
    total_labels = []
    for path_prefix in directories:
        path_prefix = basePath = path_prefix
        print(path_prefix)
        for filename in os.listdir(path_prefix):#_IA1_2'):  # You may need to specify the directory path here
            if "Dashboard" in filename:
                user, userrun = filename.split("_")[4], filename.split("_")[6]
                # Load embeddings and labels for Dashboard view
                dashboard_embeddings = np.load(path_prefix + "/" + filename)['embeddings']
                print(np.load(path_prefix + "/" + filename))
                dashboard_labels = np.load(path_prefix + "/" + filename)['labels']
                try:
                    rearview_embeddings = np.load((path_prefix + "/" + filename).replace("Dashboard", "Rear_view"))['embeddings']
                except:
                    print("Rearview")
                    rearview_embeddings = np.load((path_prefix + "/" + filename).replace("Dashboard", "Rearview"))['embeddings']
                side_embeddings = np.load((path_prefix + "/" + filename).replace("Dashboard", "Right_side_window"))['embeddings']
                minimal_length = min([len(dashboard_embeddings), len(dashboard_labels), len(rearview_embeddings), len(side_embeddings)])
                total_dashboard_embeddings.append(dashboard_embeddings[:minimal_length])
                total_labels.append(dashboard_labels[:minimal_length])
                # Load embeddings for Rear view
                total_rearview_embeddings.append(rearview_embeddings[:minimal_length])
                # Load embeddings for Side view
                total_side_embeddings.append(side_embeddings[:minimal_length])

    # Concatenate the lists to form arrays
    total_dashboard_embeddings = np.concatenate(total_dashboard_embeddings, axis=0)
    total_rearview_embeddings = np.concatenate(total_rearview_embeddings, axis=0)
    total_side_embeddings = np.concatenate(total_side_embeddings, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)

    # Save combined embeddings and labels into .npz files
    np.savez(f'clipData/{saveFolder}/total_dashboard_data.npz', embeddings=total_dashboard_embeddings)
    np.savez(f'clipData/{saveFolder}/total_rearview_data.npz', embeddings=total_rearview_embeddings)
    np.savez(f'clipData/{saveFolder}/total_side_data.npz', embeddings=total_side_embeddings)
    np.savez(f'clipData/{saveFolder}/total_labels.npz', labels=total_labels)


import argparse

parser = argparse.ArgumentParser(description='Choose test directory.')
parser.add_argument('test_dir', type=int, help='An integer for the test directory')
args = parser.parse_args()

allDirectories = ['clip_embeddingsA1_1', 'clip_embeddingsA1_2', 'clip_embeddingsA1_3', 'clip_embeddingsA1_4', 'clip_embeddingsA1_5', 'clip_embeddingsA1_6', 'clip_embeddingsA1_7']
if args.test_dir == 10:
    trainDirectories = allDirectories
    testDirectories = []
else:
    testDirectories = [f'clip_embeddingsA1_{args.test_dir}']
    trainDirectories = [dir for dir in allDirectories if dir not in testDirectories]

print(f'testDirectories = {testDirectories}')
print(f'trainDirectories = {trainDirectories}')
basePath = '/home/cvrr/Desktop/CVPRchallenge/CVPRchallenge/'

gatherTrain(trainDirectories, 'dataset')  # train=dataset, test=trainTest
if testDirectories:
    gatherTrain(testDirectories, 'trainTest')  # train=dataset, test=trainTest