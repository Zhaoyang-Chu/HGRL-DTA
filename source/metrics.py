import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import precision_recall_curve, auc


def get_cindex(Y, P):
    return concordance_index(Y, P)


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


# Prepare for rm2
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


# Prepare for rm2
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)


# Prepare for rm2
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def get_aupr(Y, P, threshold):
    Y = np.array(Y)
    P = np.array(P)
    Y = np.where(Y >= threshold, 1, 0)
    P = np.where(P >= threshold, 1, 0)
    precision, recall, _ = precision_recall_curve(Y, P)
    aupr = auc(recall, precision)
    return aupr


def get_pearson(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.corrcoef(Y, P)[0, 1]


def model_evaluate(Y, P, dataset):
    thresholds = {"davis": 7.0, "kiba": 12.1}
    return (get_mse(Y, P), get_cindex(Y, P), get_rm2(Y, P), get_pearson(Y, P), get_aupr(Y, P, thresholds[dataset]))


if __name__ == '__main__':
    G = [5.0, 7.251812, 5.0, 7.2676063, 5.0, 8.2218485, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.7212462, 5.0, 5.0, 5.0, 5.0, 5.0, 6.4436975, 5.0, 5.0, 5.60206, 5.0, 5.0, 5.1426673, 5.0, 5.0, 6.387216, 5.0, 5.0, 5.0, 6.251812, 5.0, 5.0, 5.0, 5.0, 5.0, 6.958607, 5.0, 5.0, 5.0, 5.0, 7.1739254, 5.0, 5.0, 5.0, 6.207608, 5.0, 5.5850267, 5.0, 6.481486, 5.0, 6.455932, 5.0, 5.0, 6.853872, 5.7212462, 5.0, 5.6575775, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.29243, 5.6382723, 5.0, 5.0, 5.0, 5.0, 5.0, 5.4317985, 5.0, 6.6777806, 5.0, 5.0, 5.0, 5.0, 5.5086384, 5.0, 5.0, 5.4436975, 5.0, 5.0, 5.6777806, 5.0, 5.075721, 5.0, 5.0, 8.327902, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    P = [5.022873, 7.0781856, 4.9978094, 6.7880363, 5.0082135, 8.301622, 5.199977, 5.031757, 5.282739, 5.1505866, 5.0371256, 5.0158253, 7.235809, 5.0488424, 5.0158954, 5.014982, 5.0353045, 5.0385847, 6.210839, 5.0246162, 5.040341, 5.9972534, 5.022253, 5.024069, 5.0325136, 5.858346, 5.1466026, 7.353938, 5.041976, 5.010902, 5.0101852, 5.7545958, 5.0263815, 5.0000725, 4.985109, 5.055313, 5.0001907, 6.8203254, 5.0954485, 5.1212735, 5.0224247, 5.0497823, 6.8255396, 5.0044026, 4.9908457, 5.0110598, 6.855809, 5.297818, 6.2044125, 5.0267057, 6.1194935, 5.005172, 5.6843953, 5.0014734, 5.0232143, 7.3333316, 5.8368444, 5.2844615, 5.8721313, 5.040511, 5.057362, 5.0058765, 5.018214, 5.0278683, 4.995488, 6.170251, 5.2143936, 5.0082054, 5.0141716, 5.560684, 5.0162783, 5.022541, 5.4540567, 5.023486, 5.0640993, 4.9965744, 5.0399494, 5.0136223, 5.1999803, 6.3908367, 5.022854, 5.0350113, 5.002722, 5.0313835, 5.175599, 5.1362724, 5.137325, 5.6480265, 5.03323, 5.054763, 8.333924, 5.0164843, 5.2512374, 5.02013, 5.023677, 5.0309353, 5.031672, 6.3660593, 5.035504, 5.0222054]
    print(model_evaluate(np.array(G), np.array(P), "davis"))
    