import pickle

def save_model(dir: str, model) -> None:
    """
    Saving model config in pkl file.
    :param dir: directory to save the model to
    :param model: model
    """

    pickle.dump(model, open(dir, 'wb'))


def load_model(dir: str):
    """
    Loading model from pkl file.
    :param dir: directory to load the model from
    :return: model
    """

    return pickle.load(open(dir, 'rb'))
