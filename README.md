# random_forest_3d_inversion
This repo includes Python codes for training a random forest classifier from training samples.
The random forest classifier takes gravity and magnetic data as inputs and predicts a density contrast for a 3d subsurface model within a defined size and dimension.

Please follow the following steps:
1. Run traning_dataset.py to generate training samples
2. Run rf_log_classifier_joint_3D.py to fit a random forest classifier
3. Run rf_log_prediction.py for prediction and testing over new data
