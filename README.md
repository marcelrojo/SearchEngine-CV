# SearchEngine-CV

The aim of this project is to take an input image and output the most similar images from our dataset.

## Dataset

The dataset used is a subset of the [Caltech 256 Image Dataset](https://www.kaggle.com/datasets/jessicali9530/caltech256/data). While the full dataset contains 257 object categories and a total of 30,607 images, this project utilizes 100 categories from the Caltech 256 dataset, along with 6 custom object categories. Each category contains approximately 100 images, resulting in a total of 11,520 images.

## Tested Models

The models tested in this project are organized within the following directories:

- Pure_ResNet: This directory contains the `pure_resnet.ipynb` file used for extracting the embedding space from the ResNet50 model, as well as the `ResNet50.keras` file, which contains the extracted model.
- Triplet_Loss_Model: This directory includes the `model_training.ipynb` file, which can be used to train an arbitrary feature extractor model with triplet loss and contrastive models. Additionally, it contains 3 models that were trained using this approach: a retrained ResNet, a custom classification model, and a model trained purely on triplet loss.
- Classification_Model: This directory contains `classification_model.ipynb` and `classification_model_final.ipynb` files, along with a custom classification model. The trained model is saved as a .keras file, and the corresponding embeddings are stored in a .csv file.
- Siamese_Model: This directory contains experiments on creating a Siamese network for calculating similarity between images.

## Utils folder

The `utils` directory contains scripts for generating CSV files with embeddings for each image in our dataset, as well as files for testing solutions. These scripts allow for displaying the most similar images based on chosen queries.

## Streamlit GUI

The `App` directory contains the necessary code for running the Streamlit app. Unfortunately, due to file size limitations, the CSV files could not be included in this directory. It is recommended to generate the CSV files using the `generate_csv` script from the `utils` directory if you wish to run the Streamlit app locally.

![Image](https://github.com/user-attachments/assets/8b2db828-b8fa-4c9a-8d6c-c367e0b8dc77)
![Image](https://github.com/user-attachments/assets/4b0d764a-cbca-4187-a3cc-25cf9767872b)

