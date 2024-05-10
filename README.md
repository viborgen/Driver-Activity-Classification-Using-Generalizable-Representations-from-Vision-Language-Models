# Driver Activity Classification Using Vision-Language Models

This codebase supports the research presented in the paper "Driver Activity Classification Using Generalizable Representations from Vision-Language Models." The processes described here involve generating, combining, and utilizing CLIP embeddings for classifying driver activities based on video data.

## Overview

The project consists of the following stages:

1. **Generate CLIP Embeddings**:
   - Generate embeddings for your desired data using the CLIP model. This stage transforms video frames into CLIP embeddings.

2. **Combine Embeddings**:
   - Combine the generated embeddings into `.npy` files.

3. **Train the Network**:
   - Load the `.npy` files containing the embeddings and use it to train a neural network.

4. **Apply Filtering**:
   - Implement filtering to smoothen the results. Given the sequential nature of video data this is effective against outliers.

## Resources

- **Generated Embeddings**: The embeddings generated from Stage A are available for public access. You can download them from Kaggle:
  [Multiview CLIP-Generated Embeddings](https://www.kaggle.com/datasets/mathiasviborg/multiview-clip-generated-embeddings)

## Getting Started

To use this codebase, clone the repository and follow the instructions in each component's respective directory. Each folder contains detailed README files that guide you through the processes involved in that stage.

```bash
git clone https://github.com/your-repository-url
cd your-repository-directory
