# Deep Learning Image Classifier

This project represents the culmination of the Udacity "AI Programming with Python" Nanodegree, showcasing an advanced image classifier using deep learning techniques. The model excels in accurately identifying and classifying images across various categories.

## Key Features

- **High Accuracy**: The image classifier harnesses the power of a Convolutional Neural Network (CNN) trained on an extensive dataset, resulting in exceptional accuracy for image classification.

- **Result Visualization**: The project provides intuitive tools for visualizing prediction results, enhancing user understanding and evaluation of classifications.

- **Modularized Python Code**: The codebase is meticulously modularized, facilitating seamless support and updates.

- **Best Practices Adherence**: The code adheres to best practices, including linting, meaningful variable naming, modularization, dependency injection, dependency separation, and comprehensive docstring documentation.

## How to Use the Python Script

1. Clone this repository to your local machine.
2. To train the model, execute the following command:

   ```bash
   python main.py --mode train --data_directory data_directory --architecture vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20

3. To predict with the model run the command:

   ```bash
   python main.py --mode predict --image_path image_path --checkpoint_path checkpoint_path --top_k 1 --category_names category_to_name.json.json

4. Include the --gpu flag to enable GPU usage during training or prediction.
5. Ensure CUDA for NVIDIA is installed if GPU is enabled.

## Dataset

The model was trained with the iris dataset of flowers.

## Credits

This project has been developed by Adrian Cuevas Tavizon as part of the Udacity "AI Programming with Python" Nanodegree.
