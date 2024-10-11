import os
import math
import datetime
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.driver_feature_model import DriverFeatureModel
from loguru import logger
from scipy.special import rel_entr


def idm_model(ego_speed, delta_speed, gap):
    s0 = 3.4968872
    t0 = 0.8485575
    b_ = 0.39194875
    a_ = 0.35958698
    v_ = 31.63398677
    beta = 2.89009438

    s = s0 + max(0.0, t0 * ego_speed - ego_speed * delta_speed / 2 / math.sqrt(a_ * b_))
    at = a_ * (1 - math.pow(ego_speed / v_, beta) - math.pow(s / gap, 2))
    return at


def draw_curve(index: int, df: pd.DataFrame, save_path: str):
    data = df.to_numpy()

    figs, axes = plt.subplots(2, 2, figsize=(20, 5))

    x = data[:, 0]

    speed_index = 1
    gap_index = 5

    # 绘制速度曲线
    ax: plt.Axes = axes[0, 0]
    real_speeds = data[:, speed_index]
    ax.plot(x, real_speeds, color="r", lw=2)
    idm_speeds = data[:, speed_index + 1]
    ax.plot(x, idm_speeds, color="b", lw=2)
    lstm_speeds = data[:, speed_index + 2]
    ax.plot(x, lstm_speeds, color="c", lw=2)
    lstm_noise_speeds = data[:, speed_index + 3]
    ax.plot(x, lstm_noise_speeds, color="m")

    # 绘制Gap曲线
    ax: plt.Axes = axes[0, 1]
    real_gaps = data[:, gap_index]
    ax.plot(x, real_gaps, color="r", lw=2)
    idm_gaps = data[:, gap_index + 1]
    ax.plot(x, idm_gaps, color="b", lw=2)
    lstm_gaps = data[:, gap_index + 2]
    ax.plot(x, lstm_gaps, color="c", lw=2)
    lstm_noise_gaps = data[:, gap_index + 3]
    ax.plot(x, lstm_noise_gaps, color="m")

    save_path = os.path.join(save_path, str(index) + ".pdf")
    plt.savefig(save_path)


def save_driver_feature_curve(model: DriverFeatureModel,
                              model_noise: DriverFeatureModel,
                              tracks: np.ndarray,
                              hwl: int,
                              device: torch.device,
                              save_curve_path: str):
    frequency = 25
    # indexes = np.random.randint(0, len(tracks), size=(100))
    indexes = [694, 1786]

    for i in indexes:
        track: np.ndarray = tracks[i]
        logger.info("handle {}, track shape: {}".format(i, track.shape))

        ego_speed_real_list, gap_real_list, delta_speed_real_list, ego_acc_real_list = [], [], [], []
        ego_speed_simu_list_idm, gap_simu_list_idm, delta_speed_simu_list_idm, ego_acc_simu_list_idm = [], [], [], []
        ego_speed_simu_list_lstm_noise, gap_simu_list_lstm_noise, delta_speed_simu_list_lstm_noise, ego_acc_simu_list_lstm_noise = [], [], [], []
        ego_speed_simu_list_lstm, gap_simu_list_lstm, delta_speed_simu_list_lstm, ego_acc_simu_list_lstm = [], [], [], []

        j = hwl

        data_lstm = track[:hwl][np.newaxis, :]
        data_lstm_noise = track[:hwl][np.newaxis, :]

        precedingSpeed = track[j - 1][3]

        gap_lstm, ego_speed_lstm = track[j - 1][:2]
        delta_speed_lstm = precedingSpeed - ego_speed_lstm

        gap_idm, ego_speed_idm = track[j - 1][:2]
        delta_speed_idm = precedingSpeed - ego_speed_idm

        gap_lstm_noise, ego_speed_lstm_noise = track[j - 1][:2]
        delta_speed_lstm_noise = precedingSpeed - ego_speed_lstm_noise

        # logger.info("gap: {}, ego speed: {}, delta speed: {}, preceding speed: {}"
        #             .format(gap, egoSpeed, delta_speed, precedingSpeed))

        while j < track.shape[0] - 1:
            # lstm
            obs = torch.tensor(data_lstm).to(device)
            acc = model(obs)[-1].detach().item()

            ego_speed_lstm += acc / frequency

            if ego_speed_lstm < 0:
                ego_speed_lstm = 0
                acc = 0

            precedingSpeed = track[j][3]
            delta_speed_t = precedingSpeed - ego_speed_lstm
            gap_lstm += (delta_speed_t + delta_speed_lstm) / 2 * 0.04
            delta_speed_lstm = delta_speed_t

            data_lstm = data_lstm[:, 1:]
            new_item = np.array([gap_lstm, ego_speed_lstm, delta_speed_lstm, precedingSpeed], dtype=np.float32).reshape(
                (1, 1, 4))
            data_lstm = np.concatenate([data_lstm, new_item], axis=1)
            ego_speed_simu_list_lstm.append(ego_speed_lstm)
            gap_simu_list_lstm.append(gap_lstm)
            delta_speed_simu_list_lstm.append(delta_speed_lstm)
            ego_acc_simu_list_lstm.append(acc)

            # lstm noise
            obs = torch.tensor(data_lstm_noise).to(device)
            acc = model_noise(obs)[-1].detach().item()

            ego_speed_lstm_noise += acc / frequency

            if ego_speed_lstm_noise < 0:
                ego_speed_lstm_noise = 0
                acc = 0

            precedingSpeed = track[j][3]
            delta_speed_t = precedingSpeed - ego_speed_lstm_noise
            gap_lstm_noise += (delta_speed_t + delta_speed_lstm_noise) / 2 * 0.04
            delta_speed_lstm_noise = delta_speed_t

            data_lstm_noise = data_lstm_noise[:, 1:]
            new_item = np.array([gap_lstm_noise, ego_speed_lstm_noise, delta_speed_lstm_noise, precedingSpeed],
                                dtype=np.float32).reshape((1, 1, 4))
            data_lstm_noise = np.concatenate([data_lstm_noise, new_item], axis=1)

            ego_speed_simu_list_lstm_noise.append(ego_speed_lstm_noise)
            gap_simu_list_lstm_noise.append(gap_lstm_noise)
            delta_speed_simu_list_lstm_noise.append(delta_speed_lstm_noise)
            ego_acc_simu_list_lstm_noise.append(acc)

            # idm
            acc = idm_model(ego_speed_idm, delta_speed_idm, gap_idm)

            ego_speed_idm += acc / frequency

            if ego_speed_idm < 0:
                ego_speed_idm = 0
                acc = 0

            precedingSpeed = track[j][3]
            delta_speed_t = precedingSpeed - ego_speed_idm
            gap_idm += (delta_speed_t + delta_speed_idm) / 2 * 0.04
            delta_speed_idm = delta_speed_t

            ego_speed_simu_list_idm.append(ego_speed_idm)
            gap_simu_list_idm.append(gap_idm)
            delta_speed_simu_list_idm.append(delta_speed_idm)
            ego_acc_simu_list_idm.append(acc)

            # real
            gap_real_list.append(track[j][0])
            ego_speed_real_list.append(track[j][1])
            delta_speed_real_list.append(track[j][2])
            ego_acc_real_list.append((track[j][1] - track[j - 1][1]) / 0.04)

            j += 1

        df = pd.DataFrame()
        ids = np.arange(0, len(ego_speed_real_list), 1)
        df[0] = ids
        df[1] = ego_speed_real_list
        df[2] = ego_speed_simu_list_idm
        df[3] = ego_speed_simu_list_lstm
        df[4] = ego_speed_simu_list_lstm_noise

        df[5] = gap_real_list
        df[6] = gap_simu_list_idm
        df[7] = gap_simu_list_lstm
        df[8] = gap_simu_list_lstm_noise

        df[9] = delta_speed_real_list
        df[10] = delta_speed_simu_list_idm
        df[11] = delta_speed_simu_list_lstm
        df[12] = delta_speed_simu_list_lstm_noise

        df[13] = ego_acc_real_list
        df[14] = ego_acc_simu_list_idm
        df[15] = ego_acc_simu_list_lstm
        df[16] = ego_acc_simu_list_lstm_noise

        csv_save_path = os.path.join(save_curve_path, str(i) + ".csv")
        df.to_csv(csv_save_path, index=False)
        draw_curve(i, df, save_curve_path)


def calcu_kl_th(tracks: np.ndarray, th_list) -> float:
    real_th_list = []
    for track in tracks:
        for data in track:
            if data[1] > 1e-8:
                th = data[0] / data[1]
                real_th_list.append(th)


    bin_edges = np.arange(-1, 28, 0.1) if tracks.shape[0] > 1000 else np.arange(-1, 99, 0.1)

    real_bin_indices = np.digitize(np.array(real_th_list), bin_edges)

    frequencies = np.bincount(real_bin_indices - 1, minlength=len(bin_edges))
    real_frequencies = frequencies / np.sum(frequencies)



    model_bin_indices = np.digitize(np.array(th_list), bin_edges)
    frequencies = np.bincount(model_bin_indices - 1, minlength=len(bin_edges))
    model_frequencies = frequencies / np.sum(frequencies)
    mask = model_frequencies > 0
    KL_divergence = np.sum(rel_entr(real_frequencies[mask], model_frequencies[mask]))
    print(real_frequencies, model_frequencies)
    return KL_divergence


def calcu_kl_ttc(tracks: np.ndarray, ttc_list) -> float:
    real_ttc_list = []
    for track in tracks:
        for data in track:
            ttc = - data[0] / data[2]
            if ttc > 0 and ttc < 100:
                real_ttc_list.append(ttc)

    bin_edges = np.arange(0, 105, 1) if tracks.shape[0] > 1000 else np.arange(-5, 105, 1)
    real_bin_indices = np.digitize(np.array(real_ttc_list), bin_edges)
    frequencies = np.bincount(real_bin_indices - 1, minlength=len(bin_edges))
    real_frequencies = frequencies / np.sum(frequencies)

    model_bin_indices = np.digitize(np.array(ttc_list), bin_edges)
    frequencies = np.bincount(model_bin_indices - 1, minlength=len(bin_edges))
    model_frequencies = frequencies / np.sum(frequencies)

    mask = model_frequencies > 0
    KL_divergence = np.sum(rel_entr(real_frequencies[mask], model_frequencies[mask]))
    return KL_divergence


def test_driver_feature_model(model: DriverFeatureModel,
                              tracks: np.ndarray,
                              hwl: int,
                              device: torch.device,
                              save_path: str):
    # mse 误差
    speedSumError = 0
    gapSumError = 0
    jerkSumError = 0
    minTTCSum = 0
    minError = 100
    maxError = -1
    minIndex = -1
    maxIndex = -1
    count = 0
    frequency = 25
    collisions = np.zeros((len(tracks)))

    speed_error_list = []
    gap_error_list = []
    ttc_list = []
    th_list = []
    jerk_list = []
    speed_list = []
    gap_list = []

    for i in range(len(tracks)):
        track: np.ndarray = tracks[i]
        logger.info("handle {}, track shape: {}".format(i, track.shape))
        data = track[:hwl][np.newaxis, :]
        j = hwl
        speedError = 0
        gapError = 0
        jerkError = 0
        minTTC = 1e8

        gap, egoSpeed, deltaSpeed, precedingSpeed = track[j - 1]
        last_acc = (track[j - 1][1] - track[j - 2][1]) / 0.04

        while j < track.shape[0] - 1:
            obs = torch.tensor(data).to(device)
            acc = model(obs)[-1].detach().item()
            egoSpeed += acc / frequency

            if egoSpeed < 0:
                egoSpeed = 0
                acc = 0

            precedingSpeed = track[j][3]
            deltaSpeed_t = precedingSpeed - egoSpeed
            gap += (deltaSpeed_t + deltaSpeed) / 2 * 0.04
            deltaSpeed = deltaSpeed_t

            if gap < 0:
                collisions[i] = 1
            data = data[:, 1:]
            new_item = np.array([gap, egoSpeed, deltaSpeed, precedingSpeed], dtype=np.float32).reshape((1, 1, 4))
            data = np.concatenate([data, new_item], axis=1)

            se = (egoSpeed - track[j][1]) ** 2
            ge = (gap - track[j][0]) ** 2
            speedError += se
            gapError += ge

            speed_error_list.append(float(se))
            gap_error_list.append(float(ge))

            ttc = float(- gap / deltaSpeed)

            if ttc > 0:
                if ttc < minTTC:
                    minTTC = ttc
                if ttc < 100:
                    ttc_list.append(ttc)

            gap_list.append(float(gap))
            th_list.append(float(gap) / max(0.0001, egoSpeed))

            last_acc = acc

            count += 1
            j += 1

        error = (speedError + gapError) / (j - hwl - 1)

        if error < minError:
            minError = error
            minIndex = i
        if error > maxError:
            maxError = error
            maxIndex = i

        speedSumError += speedError / (track.shape[0] - 1)
        gapSumError += gapError / (track.shape[0] - 1)
        jerkSumError += jerkError / (track.shape[0] - 1)
        minTTCSum += minTTC

    kl_th = calcu_kl_th(tracks, th_list)
    kl_ttc = calcu_kl_ttc(tracks, ttc_list)

    logger.info("gap mse: {}, jerk mae: {}, collision count: {}, min ttc: {}, kl ttc: {}, kl th: {}"
    .format(
        gapSumError / len(tracks),
        jerkSumError / len(tracks),
        np.sum(collisions),
        minTTCSum / len(tracks),
        kl_ttc, kl_th
    ))

    # error_list = np.array([speed_error_list, gap_error_list, ttc_list, jerk_list, speed_list, gap_list])
    # np.save(save_path, error_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model_path", type=str, default=None)
    parser.add_argument('-lp', '--log_path', default="/home/public/lzg/car-following-model/log", type=str)
    parser.add_argument("-track", "--track_path", type=str, default=None)
    parser.add_argument("-hwl", "--history_windows_length", type=int, default=25)
    parser.add_argument("-mse", "--mse", action='store_true')
    args = parser.parse_args()
    run_name = os.path.basename(__file__).split(".")[0]
    log_path = args.log_path + os.sep + run_name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.add(log_path)
    logger.info(args.__str__())
    if args.model_path is None or args.track_path is None:
        exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # jit_model = torch.jit.load(args.model_path)
    model = DriverFeatureModel(4, 200, 2)
    model.load_state_dict(torch.load(args.model_path))
    # for param, target_param in zip(jit_model.parameters(), model.parameters()):
    #     target_param.dataset.copy_(param.dataset)
    model.to(device)

    tracks = np.load(args.track_path).transpose([0, 2, 1]).astype(np.float32)
    # tracks = tracks[:10]
    hwl = args.history_windows_length

    if args.mse:
        save_path = args.model_path + "-" + os.path.basename(args.track_path) + "-error.npy"
        test_driver_feature_model(model, tracks, hwl, device, save_path)
