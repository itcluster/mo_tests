import numpy as np
import matplotlib.pyplot as plt
def plot_feature_importances_cancer(model, data_):
    n_features = data_.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_.feature_names)
    plt.xlabel("character's necessity")
    plt.show()
