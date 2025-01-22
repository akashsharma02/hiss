import h5py
import os
import argparse
import einops

import numpy as np
from tqdm import tqdm
import torch
import pytorch_kinematics as pk

import sys

sys.path.append("./")

from hiss.utils.data_utils import (
    aggregate_list_of_dicts,
    get_data_path,
    get_demo_dirs,
)
from hiss.utils.data_utils import DATA_FILENAME, DICT_KEY

XELA_FLATTEN_ORDER = {
    "3aftc_palm_link": 30,
    "link_15_4x4_palm_link": 16,
    "link_14_4x4_palm_link": 16,
    "0aftc_palm_link": 30,
    "link_2_4x4_palm_link": 16,
    "link_1A_4x4_palm_link": 16,
    "link_1B_4x4_palm_link": 16,
    "1aftc_palm_link": 30,
    "link_6_4x4_palm_link": 16,
    "link_5A_4x4_palm_link": 16,
    "link_5B_4x4_palm_link": 16,
    "2aftc_palm_link": 30,
    "link_10_4x4_palm_link": 16,
    "link_9A_4x4_palm_link": 16,
    "link_9B_4x4_palm_link": 16,
    "ahr_palm_2_4x6_palm_link": 24,
    "ahr_palm_1_4x6_palm_link": 24,
    "ahr_palm_3_4x6_palm_link": 24,
}


def get_sensor_grid(patch_name):
    if "aftc" in patch_name:
        h, w, d = 0.031, 0.039, 0.029  # numbers taken from mesh boundingbox
        h_res, w_res = 6, 6
        x = (
            np.linspace(0.5 - h_res / 2, h_res / 2 + 0.5, h_res, endpoint=False)
            * h
            / h_res
        )
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx_, yy_ = np.meshgrid(x, y)
        xx = np.concatenate(
            [xx_[:4, :].flatten(), xx_[-2, 1:-1], xx_[-1, 2:-2]], axis=0
        )
        yy = np.concatenate(
            [yy_[:4, :].flatten(), yy_[-2, 1:-1], yy_[-1, 2:-2]], axis=0
        )
    elif "4x4" in patch_name:
        h, w, d = 0.026, 0.024, 0.0044  # numbers taken from mesh boundingbox
        h_res, w_res = 4, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
    elif "4x6" in patch_name:
        h, w, d = 0.052, 0.032, 0.0  # numbers taken from mesh boundingbox
        h_res, w_res = 6, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
    return xx, yy, d


def joint_angles_to_poses(xela_kinematic_chain, joint_angles: np.ndarray):
    joint_angles = torch.tensor(joint_angles)
    joint_poses = xela_kinematic_chain.forward_kinematics(joint_angles)
    poses = []
    for k, num_sensors in XELA_FLATTEN_ORDER.items():
        joint_pose = joint_poses[k].get_matrix().numpy()
        xx, yy, d = get_sensor_grid(k)
        sensor_positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        sensor_positions = np.concatenate(
            [sensor_positions, np.zeros_like(sensor_positions)], axis=-1
        )
        sensor_positions[..., -2] = d
        sensor_positions[..., -1] = 1
        t = joint_pose.shape[0]
        joint_pose = einops.repeat(joint_pose, "t i j -> t s i j", s=num_sensors)
        joint_pose = einops.rearrange(joint_pose, "t s i j -> (t s) i j")
        sensor_positions = einops.repeat(sensor_positions, "s c -> (t s) c", t=t)
        sensor_pose = np.einsum("m i j, m j -> m i", joint_pose, sensor_positions)
        joint_pose[..., :, 3] = sensor_pose
        joint_pose = einops.rearrange(joint_pose, "(t s) i j -> t s i j", s=num_sensors)
        poses.append(joint_pose)
    poses = np.concatenate(poses, axis=1)
    positions = poses[..., :3, 3]
    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process demo data: This script allows you to process the \
            raw data into a single h5 file"
    )
    parser.add_argument(
        "--dataset-dir",
        "-dd",
        type=str,
        default="joystick_control_hiss_dataset",
        help="Directory containing the dataset to process",
    )
    parser.add_argument(
        "--data-suffix",
        "-ds",
        type=str,
        default="processed",
        help="Suffix to append to the processed data file",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="/home/akashsharma/workspace/datasets/joystick_control_hiss_dataset/urdf/ahrcpcpn.urdf",
        help="Path to the URDF file",
    )
    parser.add_argument(
        "--vis", "-v", action="store_true", default=False, help="Visualize data"
    )
    args = parser.parse_args()
    VIS = args.vis
    urdf_path = args.urdf_path
    xela_kinematic_chain = pk.build_chain_from_urdf(open(urdf_path).read())

    # List modalities to process.
    modalities = ["xela", "extreme3d", "xela_sensor_pos"]
    demo_dirs = get_demo_dirs(args.dataset_dir)
    proc_data_path = get_data_path(args.dataset_dir, args.data_suffix)
    print(f"Writing to: {proc_data_path}")

    dur_list = []
    data_list = []
    freq_lists = {m: [] for m in modalities}
    mod_eids = {m: [0] for m in modalities}
    curr_mod_eid = {m: 0 for m in modalities}

    for demo_dir in tqdm(sorted(demo_dirs)):
        data = {}
        dur_curr = []
        for m in modalities:
            try:
                with h5py.File(
                    os.path.join(demo_dir, f"{DATA_FILENAME[m]}"), "r"
                ) as hf:
                    values = np.array(DICT_KEY[m](hf))
                    timestamps = np.array(hf["timestamps"])
                if m == "xela":
                    baseline = np.median(values[:100], axis=0, keepdims=True)
                try:
                    assert timestamps.shape[0] == values.shape[0]
                except AssertionError:
                    clip_len = min(timestamps.shape[0], values.shape[0])
                    timestamps = timestamps[:clip_len]
                    values = values[:clip_len]

                indices = np.arange(len(values))
                data[m] = values
                data[f"{m}_timestamps"] = timestamps
                data[f"{m}_indices"] = indices

                freq = 1 / np.mean(np.diff(timestamps))
                freq_lists[m].append(freq)
                if m == "xela":
                    data[m] -= baseline
                    data[m] = np.clip(data[m], -1000, 1000)

                if m == "xela_sensor_pos":
                    sensor_poses = joint_angles_to_poses(xela_kinematic_chain, values)
                    sensor_poses = einops.rearrange(sensor_poses, "t n c -> t (n c)")
                    data[m] = sensor_poses
                dur_curr.append(np.around(timestamps[-1] - timestamps[0]))
                if m == "extreme3d":
                    if min(dur_curr) < 15.0 or max(dur_curr) > 90.0:
                        # print(demo_dir, dur_curr)
                        data = {}
                        break
                    dur_list.append(dur_curr)

            except FileNotFoundError as e:
                print(e)
                # break

        if len(data) > 0:
            data_list.append(data)

            curr_mod_eid = {m: curr_mod_eid[m] + len(data[m]) for m in modalities}
            for m in modalities:
                mod_eids[m].append(curr_mod_eid[m])
            done_list = [True] * len(data_list)
    for m in freq_lists:
        print(f"{m}: {np.mean(freq_lists[m])}")

    pardir = os.path.abspath(os.path.join(proc_data_path, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    data_list = aggregate_list_of_dicts(data_list)
    xela_data = np.array(data["xela"])
    xela_data = xela_data.reshape(xela_data.shape[0], -1, 3)
    xela_data = xela_data.reshape(-1, 3)

    # Visualize Xela data histogram
    if VIS:
        import matplotlib.pyplot as plt

        ax0 = plt.subplot(1, 3, 1)

        ax0.hist(xela_data[:, 0], bins=100)
        ax0.set_yscale("log")
        ax1 = plt.subplot(1, 3, 2)
        ax1.hist(xela_data[:, 1], bins=100)
        ax1.set_yscale("log")
        ax2 = plt.subplot(1, 3, 3)
        ax2.hist(xela_data[:, 2], bins=100)
        ax2.set_yscale("log")
        plt.suptitle("Xela Data Histogram")
        plt.show()

    for m in modalities:
        data_list[f"{m}_episode_ids"] = mod_eids[m]
    with h5py.File(proc_data_path, "w") as hf:
        for key, value in data_list.items():
            print(key, len(value))
            hf.create_dataset(key, data=value)
