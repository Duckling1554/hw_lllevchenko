from model.logistic_regression import predict
from conf.conf import logging
from conf.conf import settings

prediction = predict(values=settings.PREDICTION.VALUES,path_to_model=settings.MODEL.PATH_SVM)
logging.info(f'prediction: {prediction}')
