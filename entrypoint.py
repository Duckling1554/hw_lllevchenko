from model.model_funcs import predict, initialize_model
from conf.conf import logging
from conf.conf import settings
import argparse
from os.path import exists

parser = argparse.ArgumentParser(description='Parser to choose model and input values')
parser.add_argument(
    '--prediction_model',
    type=str,
    choices=['SVM', 'LOG_REG'],
    default='LOG_REG',
    help='Choose a model for your predictions'
)
parser.add_argument(
    '--prediction_values',
    type=str,
    default=settings.PREDICTION.VALUES,
    help=f'Input {settings.PREDICTION.NUMBER_VALUES} values separated with commas (!no spaces!)'
)
my_namespace = parser.parse_args()

if not (my_namespace.prediction_model in settings.MODEL and exists(settings.MODEL[my_namespace.prediction_model])): #
    initialize_model(model_name=my_namespace.prediction_model)

values = [float(s.strip()) for s in my_namespace.prediction_values.split(",")]
if len(values) != settings.PREDICTION.NUMBER_VALUES:
    raise Exception(f'The model is expecting {settings.PREDICTION.NUMBER_VALUES} features as input.')
path_to_model = settings.MODEL[my_namespace.prediction_model]

prediction = predict(values=[values]
                     , path_to_model=path_to_model)
logging.info(f'Prediction: {prediction}')