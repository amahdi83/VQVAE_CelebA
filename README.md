# **VQ-VAE Image Generation on CelebA Dataset** 🎨

This repository implements a Vector Quantized Variational Autoencoder (VQ-VAE) for learning and generating images from the CelebA dataset. The model is built using **PyTorch** and incorporates a residual stack for the encoder/decoder, as well as vector quantization with both the standard and EMA methods.

## **Table of Contents** 📑
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Model Architecture](#model-architecture)
7. [Customization](#customization)
8. [Acknowledgements](#acknowledgements)

---

## **Project Overview** 🚀

This project focuses on training a VQ-VAE on the CelebA dataset. VQ-VAEs are powerful for learning discrete latent representations of images and can be used for high-quality image generation tasks. The goal of this implementation is to encode images into discrete latent codes using a vector quantization approach and reconstruct images from those codes.

---

## **Features** ✨

- **Residual Networks**: For enhanced encoding and decoding.
- **Vector Quantization**: Standard and EMA versions for efficient discrete representations.
- **Custom DataLoader**: Load specific train, validation, and test datasets.
- **Custom Normalization**: Normalization function to rescale image tensors between [-1, 1].
- **CelebA Dataset Handling**: Data pre-processing and augmentation for CelebA.
  
---

## **Installation** 💻

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/your-username/vq-vae-celeba.git
cd vq-vae-celeba```

### **Step 2: Set Up a Virtual Environment (optional but recommended)**

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### **Step 3: Install Dependencies**

Make sure you have Python 3.7+ installed, then install the necessary dependencies:

```bash
pip install torch torchvision numpy matplotlib Pillow
```
---

## **Usage 🎮**

### **Step 1: Download CelebA Dataset**

Download the CelebA dataset from here and place the images in the directory structure:

```
celeba_gan/
  ├── img_align_celeba/
  ├── list_eval_partition.txt
```

### **Step 3: Training the Model**
To train the model, simply run the notebook.

---

## **Results 📊**
Here are some sample outputs after training the model on the CelebA dataset:

*Reconstructed Images:*

*Generated Images:*

---

## **Model Architecture 🏗️**

The model consists of:

*Encoder:* Maps input images to discrete latent codes.
*Decoder:* Reconstructs the input images from the discrete latent codes.
*Vector Quantizer:* Maps latent vectors to discrete embeddings.

We use two types of vector quantization:

*Vector Quantizer:* Basic version of VQ.
*Vector Quantizer EMA:* Uses exponential moving average to update embedding vectors.

---

## **Customization 🔧**

*Image Size:* Modify the img_size variable in the script to change the input image resolution.
*Model Hyperparameters:* You can change the architecture's parameters, such as num_hiddens, embedding_dim, and num_residual_layers to experiment with different VQ-VAE configurations.
*Training Parameters:* Adjust n_epochs, learning_rate, and batch_size to tune the training process.

For more advanced customization, edit the respective model classes or the training loop in train.py.

---

## **Acknowledgements 🙌**

This project uses the CelebA dataset, provided by MMLab.
The architecture is inspired by the paper "Neural Discrete Representation Learning".
Thanks to PyTorch for providing an excellent deep learning framework.
