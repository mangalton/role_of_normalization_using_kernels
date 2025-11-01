# Role Of Normalization Using Kernels

## 1. Overview

This project provides a comprehensive analysis of the impact of different data normalization techniques on the performance of Support Vector Machine (SVM) classifiers. The experiments are conducted using three different SVM kernels (Linear, RBF, and Polynomial) on two popular image classification datasets:

* **Fashion-MNIST:** A dataset of 28x28 grayscale images of clothing items.
* **CIFAR-10:** A dataset of 32x32 color images across 10 classes.

The goal is to systematically measure and visualize how scaling data affects model accuracy, training time, and other metrics to determine the optimal preprocessing pipeline for each dataset and kernel.

The project is divided into two main parts for each dataset:

1.  **Training & Analysis:** Running a full experiment to compare all combinations of kernels and normalization methods.
2.  **Inference:** Using the best-performing saved model and scaler to make predictions on new, unseen images.

---

## 2. Key Analyses Performed

### Kernels Compared:
* `linear`: Linear SVM
* `rbf`: Radial Basis Function (Gaussian) Kernel
* `poly`: Polynomial Kernel

### Normalization Methods Tested:
* `none`: Using the raw, unscaled pixel data.
* `standard`: `StandardScaler` (zero mean, unit variance).
* `minmax`: `MinMaxScaler` (scales data to a [0, 1] range).
* `robust`: `RobustScaler` (uses median and IQR, robust to outliers).
* `l2`: `L2 Normalization` (scales each sample to have a unit norm).

### Metrics Tracked:
* Train & Test Accuracy
* Training Time
* Per-Class Accuracy (Precision, Recall, F1-Score)
* Confusion Matrix
* Number of Support Vectors

---

## 3. Project File Structure

Here is a description of each file and its role in the project.

### `Role_of_normalization_MNISTFashion.py`
* **Purpose:** The main training and analysis script for the **Fashion-MNIST** dataset.
* **Function:**
    * Loads the Fashion-MNIST data.
    * Flattens the 28x28 images into 784-feature vectors.
    * Systematically trains and evaluates an SVM for all 15 combinations (5 normalizations x 3 kernels).
    * Generates a results DataFrame and plots to compare performance.
    * Conducts a detailed analysis (confusion matrix, classification report) of the **best** performing model.
    * Analyzes and plots misclassified samples.

### `Role_of_normalization_CIFAR-10.py`
* **Purpose:** The main training and analysis script for the **CIFAR-10** dataset.
* **Function:**
    * Loads the CIFAR-10 data.
    * Flattens the 32x32x3 color images into 3072-feature vectors.
    * Performs the same 15-combination experiment as the Fashion-MNIST script.
    * Saves the best-performing model (`best_svm_model.joblib`) and its corresponding scaler (`best_scaler.joblib`) to disk.
    * Includes a hyperparameter tuning section for the best kernel.

### `InferenceForMNISTFashionDataset.py`
* **Purpose:** A script to run predictions using a pre-trained **Fashion-MNIST** model.
* **Function:**
    * Loads a saved model (`.joblib`) and scaler (`.joblib`).
    * Loads the fresh Fashion-MNIST test set.
    * Takes a few sample images, applies the saved scaler, and feeds them to the saved model.
    * Visualizes the image and prints the model's prediction vs. the true label.

### `InferenceForCIFAR-10Dataset.py`
* **Purpose:** A script to run predictions using the pre-trained **CIFAR-10** model.
* **Function:**
    * Loads `best_svm_model.joblib` and `best_scaler.joblib` (saved by `Role_of_normalization_CIFAR-10.py`).
    * Loads the fresh CIFAR-10 test set.
    * Picks 5 random images, applies the preprocessing pipeline, and predicts their class.
    * Displays the image, prediction, and true label.

---

## 4. How to Run the Project

###  Requirements
You will need the following Python libraries. You can install them using `pip`:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow pandas
```
### 1. Training and Analysis

To run the full experiments, execute the main analysis scripts from your terminal.

Note: These scripts train 15 separate SVM models on large datasets and may take significant time to complete.

###For Fashion-MNIST:

```bash
python Role_of_normalization_MNISTFashion.py
```

This will run the full analysis and display the comparison plots for Fashion-MNIST.

### For CIFAR-10:

```bash 
python Role_of_normalization_CIFAR-10.py
```

This will run the full analysis for CIFAR-10, display plots, and save two files:
```
best_svm_model.joblib
best_scaler.joblib
```

### 2. Running Inference

After training, you can use the inference scripts to test the saved models.


For Fashion-MNIST:

Save your desired model and scaler from the `Role_of_normalization_MNISTFashion.py experiment`.

Open `InferenceForMNISTFashionDataset.py`.

Update `model_filename` and `scaler_filename` to point to your saved files.

Run:
```bash
python InferenceForMNISTFashionDataset.py
```
### For CIFAR-10:

Run `Role_of_normalization_CIFAR-10.py` to generate `best_svm_model.joblib` and `best_scaler.joblib`.

Open `InferenceForCIFAR-10Dataset.py`.

Ensure file paths match (same directory by default).

Run:
```bash
python InferenceForCIFAR-10Dataset.py
```
