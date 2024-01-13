# CompCars Multi-Task Classification Model

We built a multi headed classification model on top of EfficientNetB5 to predict the make ID and model ID of cars for NTU CZ4042.

# Dataset
The original dataset can be obtained [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/).
We encoded labels and converted the orginal data to TFRecord files to facilitate training on TPU. Our dataset can be found on Kaggle [here](https://www.kaggle.com/adithyaxx/the-comprehensive-cars-compcars-dataset/activity).

# Trained Models
The following models can be found [here](https://drive.google.com/drive/folders/1jl7ZE9dXdwymx2dm-QLke46ZfbFXw5s9?usp=sharing).

- `cce_loss.h5` - Trained with categorical crossentropy loss
- `cce_loss_optimised.h5` - Trained with categorical crossentropy loss and optimised hyperparameters
- `focal_loss.h5` - Trained with focal loss and semi optimised hyperparameters

# Testing
The models were trained with TPU and hence, any testing can be done by running `test.ipynb` from the link above.
