# Part B Documentation — Neural Benchmark

## 1. Dataset
The benchmark utilizes the real-world **Breast Cancer Wisconsin (Diagnostic)** dataset from `sklearn.datasets`.
*(Screenshot: `documentation/screenshots/part-b/part-b-main.png`)*

## 2. Preprocessing
The dataset undergoes two critical preprocessing steps in `backend/main.py`:
1. It is split into training and testing sets using `train_test_split(stratify=y)`, ensuring both sets contain a representative proportion of malignant and benign samples.
2. The features are standardized to zero mean and unit variance using `StandardScaler`.

## 3. Neural-Network Architecture
The model is a custom Multi-Layer Perceptron (MLP) implemented entirely from scratch in pure NumPy (`backend/mlp.py`).
- **Input Layer**: Maps to the dataset feature count.
- **Hidden Layer 1**: Dense(16) $\to$ ReLU activation.
- **Hidden Layer 2**: Dense(8) $\to$ ReLU activation.
- **Output Layer**: Dense(1) $\to$ Sigmoid activation.
- **Initialization**: He Initialization ($W \sim \mathcal{N}(0, \sqrt{2/\text{fan\_in}})$).

## 4. Input Features
The dataset contains exactly 30 numerical features derived from digitized images of fine needle aspirates (FNA) of breast masses.

## 5. Training/Test Split
An 80/20 split is utilized, resulting in exactly **455 Train Samples** and **114 Test Samples**.

## 6. Optimizers
All 7 stateful optimizers implemented in Part A (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW) are available to optimize the MLP's weights.

## 7. Learning Rate
The base learning rate $\eta$ is configurable via the UI, dictating the step size applied to the gradients during backpropagation.

## 8. Epochs
An epoch represents one complete pass of the entire training dataset through the neural network. Users can configure the total number of epochs (default 100).
*(Screenshot: `documentation/screenshots/part-b/part-b-training-config.png`)*

## 9. Batch Size
The benchmark uses mini-batch gradient descent. The dataset is shuffled at the start of each epoch and sliced into batches of the specified size (default 32). This introduces stochasticity into the gradient calculation, helping the model escape local minima.

## 10. Loss Function
The model optimizes for **Binary Cross Entropy (BCE)**, which measures the distance between the model's predicted probabilities (0 to 1) and the true binary labels.

## 11. Training Procedure
1. Shuffle dataset indices.
2. Slice into mini-batches.
3. Forward pass batch through MLP.
4. Calculate BCE loss and gradients via manual backpropagation chain rule.
5. Flatten gradients and pass to optimizer for step update.
6. Unflatten updated weights and apply to MLP.

## 12. Validation Procedure
Because this is an educational tool, the "Validation" metrics displayed in the UI actually represent performance on the held-out 20% Test set at the end of each epoch. It is not used for hyperparameter tuning.

## 13. Test Procedure
At the end of each epoch, the model runs a forward pass on the `X_test_scaled` dataset without calculating gradients. The resulting probabilities are thresholded at 0.5 to determine classification accuracy.

## 14. Training Loss
The Training Loss graph (`documentation/screenshots/part-b/part-b-training-loss.png`) plots the BCE calculated on the training data. It demonstrates how well the optimizer is fitting the model to the data it has seen.

## 15. Validation Loss
The Validation (Test) Loss graph (`documentation/screenshots/part-b/part-b-validation-loss.png`) plots BCE on unseen data. Divergence between training loss (going down) and validation loss (going up) visually demonstrates overfitting.

## 16. Test Accuracy
The Test Accuracy graph (`documentation/screenshots/part-b/part-b-test-accuracy.png`) shows the percentage of the 114 test samples correctly classified by the model over time.

## 17. Convergence Epoch
Calculated retroactively by the backend. It represents the *first* epoch where the Test Loss enters and strictly remains within a 1% tolerance window of the final epoch's Test Loss. It provides a concrete numerical metric for convergence speed.

## 18. Benchmark Comparison Table
Upon animation completion, a summary table appears detailing the final metrics across all selected optimizers for objective comparison.
*(Screenshot: `documentation/screenshots/part-b/part-b-results-table.png`)*
