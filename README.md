
# Emergency vs Non-Emergency Vehicle Classification

This project implements a deep learning model to classify vehicles into emergency and non-emergency categories. The model utilizes techniques like Batch Normalization and Dropout to improve performance and generalization.

## Project Structure

- `data/Dataset`: Contains the dataset including the `emergency_classification.csv` file and the `images` directory.
- `notebook`: Contains Jupyter notebooks for various techniques like batch normalization, dropout, early stopping, etc.
- `source`: Contains Python scripts corresponding to the Jupyter notebooks.
- `requirements.txt`: Lists all the dependencies for the project.

## Getting Started

To get started with this project, first clone the repository and then install the required dependencies:

```
pip install -r requirements.txt
```

### Steps in the Project

1. **Loading the dataset**: The dataset is loaded from the `data/Dataset` directory.
2. **Pre-processing the data**: Images are pre-processed and normalized.
3. **Creating training and validation set**: The dataset is split into training and validation sets.
4. **Defining the model architecture**: The model architecture is defined with Keras, utilizing layers like `Dense` and `InputLayer`.
5. **Compiling the model**: The model is compiled using the Adam optimizer.
6. **Training the model**: The model is trained on the dataset with techniques like Batch Normalization and Dropout.
7. **Evaluating model performance**: The model's performance is evaluated on the validation set.

### Model Techniques

- `batchnormalization.ipynb`: Implements batch normalization.
- `dropoutnueralnw.ipynb`: Implements dropout in a neural network.
- `earlystopping.ipynb`: Implements early stopping.
- `gradientclipping.ipynb`: Implements gradient clipping.
- `imageaugmentation.ipynb`: Implements image augmentation.
- `imagegeneratorfitgenerator.ipynb`: Implements image data generation and fitting.
- `modelcheck.ipynb`: Implements model checkpointing.
- `weightinitialization.ipynb`: Implements weight initialization techniques.

## Usage

To train the model, navigate to the `notebook` directory and open the Jupyter notebooks. Each notebook is self-contained and includes steps for different techniques applied to the deep learning model.

```
cd notebook
jupyter notebook
```

Select the notebook you wish to run, and execute the cells in order to train the model and evaluate its performance.

## Requirements

- NumPy: `1.19.5`
- Pandas: `1.3.3`
- Matplotlib: `3.4.3`
- Keras: `2.6.0`
- Scikit-learn: `0.24.2`

Ensure you have the correct versions of these libraries to avoid any compatibility issues.

## Contributing

Feel free to fork the project, submit pull requests, or send suggestions to improve the models or techniques used.
```

