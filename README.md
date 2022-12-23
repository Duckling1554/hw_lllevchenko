# hw_lllevchenko

## Models
In the current application 2 models are implemented: SVM and Logistic Regression. 
The base of the model was taken from: https://github.com/5x12/ml-cookbook/tree/master/classification.

The models were trained using the dataset: https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv.

## Predicting

To get prediction, run `python3 entrypoint.py --prediction_model <name of the model> --prediction_values <values separated by comma (no spaces)>`. 
You should input exact number (13 by default) of values as input for prediction. To get more info about input, run `python3 entrypoint.py -h`.