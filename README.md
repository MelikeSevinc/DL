# Fish Species Classification Project with Deep Learning
This project aims to develop a model to classify fish species using deep learning. The model performs the learning process using images of different fish species and can then be used to identify these species.
## Dataset
This project uses a fish classification dataset obtained from Kaggle. [Details of the dataset and download link can be found here.](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
## Model Architecture
The project utilizes a deep learning model comprised of the following fundamental layers:
Input Layer: A rescaling layer scales the pixel values of input images to the range [0, 1].
Flatten Layer: Flattens the input images, making them suitable for multilayer perceptrons.
Hidden Layers: One or two Dense layers, each consisting of 128 nodes and using the 'relu' activation function.
Dropout Layer: Added with a 20% dropout rate to prevent overfitting.
Output Layer: Contains a Dense layer with a softmax activation function to classify 9 different fish species.
## Model Creation Function
The model is created using the create_model function, which allows for customization of the number of layers, units per layer, dropout rate, and optimizer. The default configuration includes two hidden layers, each with 128 units and a dropout rate of 30%. The Adam optimizer is used for compiling the model.
## Training the Model
The model is trained using a training dataset split from the original dataset, and it is validated using a separate validation set. The training process includes monitoring metrics such as accuracy, loss, precision, and recall to ensure the model is learning effectively.
## Results
Loss graph: This graph visualizes the model's error during training. A decreasing loss indicates that the model is learning effectively.
Accuracy graph: This graph shows how accurately the model is able to classify the data. A higher accuracy indicates better performance.
Recall and Precision graphs: These graphs evaluate the model's ability to correctly identify positive instances. Recall measures the proportion of actual positives that were identified correctly, while precision measures the proportion of predicted positives that were actually correct.

You can view and try the project using the Kaggle link here: [DL-FishClassification](https://www.kaggle.com/code/melikesevin/dl-fishclassification)
