import os
import math
import numpy as np
import argparse
import open3d as o3d
import MinkowskiEngine as ME
import torch
import typing as t
import util.transform_estimation as te

from urllib.request import urlretrieve
from model.resunet import ResUNetBN2C
from util.visualization import get_colored_point_cloud_feature
from lib.eval import find_nn_gpu

if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
    print('Downloading weights...')
    urlretrieve(
        "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
        'ResUNetBN2C-16feat-3conv.pth')

if not os.path.isfile('redkitchen-20.ply'):
    print('Downloading a mesh...')
    urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/redkitchen-20.ply",
                'redkitchen-20.ply')

NN_MAX_N = 500
SUBSAMPLE_SIZE = 10000


def points_to_pointcloud(
        points: np.array, voxel_size: float = 0.025, scalars: t.Optional[np.array] = None
) -> o3d.geometry.PointCloud():
    """ convert numpy array points to open3d.PointCloud
    :param points: np.ndarray of shape (N, 3) representing floating point coordinates
    :param voxel_size: float
    :param scalars: (optional) np.ndarray of shape (N, 1), scalar of each point (e.g. FDI)
    :return: open3d.PointCloud
    """
    radius_normal = voxel_size * 2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd.estimate_covariances(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    if scalars is not None:
        colors = np.asarray([int_to_rgb(i) for i in scalars])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def int_to_rgb(val: int, min_val: int = 11, max_val: int = 48, norm: bool = True):
    if val > max_val:
        raise ValueError("val must not be greater than max_val")
    if val < 0 or max_val < 0:
        raise ValueError("arguments may not be negative")
    if val < min_val:
        raise ValueError("val must be greater than min_val")

    i = (val - min_val) * 255 / (max_val - min_val)
    r = round(math.sin(0.024 * i + 0) * 127 + 128)
    g = round(math.sin(0.024 * i + 2) * 127 + 128)
    b = round(math.sin(0.024 * i + 4) * 127 + 128)
    if norm:
        r /= 255
        g /= 255
        b /= 255
    return [r, g, b]


def draw_tensor_points(
        tensors: t.List[t.Union[np.ndarray, torch.Tensor]],
        extract_points: bool = False,
        uniform_color: bool = True,
        color_arrays: t.Optional[t.List[t.Union[np.ndarray, list]]] = None,
        min_color: t.Optional[float] = 11,
        max_color: t.Optional[float] = 48,
) -> None:
    """
    :param tensors: list of tensor, either numpy or torch with each having a shape of (N, 3) or (N, 4)
    :param extract_points: bool, to extract the points (N, 0:3)
    :param uniform_color: bool, apply uniform color
    :param color_arrays: (optional) list of np.ndarray or list of shape (N, 1) that contains point color
    :param min_color: (optional) float of min color for colormap
    :param max_color: (optional) float of max color for colormap
    :return:
    """
    if not isinstance(tensors, list):
        tensors = [tensors]
    if not isinstance(color_arrays, list):
        color_arrays = [color_arrays]

    if len(color_arrays) < len(tensors):
        color_arrays += [None] * (len(tensors) - len(color_arrays))

    pcd_list = []
    for tt, ca in zip(tensors, color_arrays):
        if isinstance(tt, torch.Tensor):
            tt = tt.clone().numpy()
        elif isinstance(tt, np.ndarray):
            tt = tt.copy()
        else:
            raise ValueError(
                "Tensor type not supported, should be torch.Tensor or np.ndarray"
            )
        np_points = np.squeeze(tt)
        if extract_points:
            np_points = np_points[:, 0:3]
        pcd_temp = points_to_pointcloud(np_points)
        if uniform_color and ca is None:
            pcd_temp.paint_uniform_color(list(np.random.uniform(size=3)))
        elif ca is not None:
            if min_color is None:
                min_color = np.min(ca)
            if max_color is None:
                max_color = np.max(ca)
            colors = np.asarray(
                [int_to_rgb(i, min_val=min_color, max_val=max_color) for i in ca]
            )
            pcd_temp.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd_temp)
    o3d.visualization.draw_geometries(pcd_list)


def apply_transformation(
        points: t.Union[np.ndarray, torch.Tensor],
        transformation: t.Union[np.ndarray, torch.Tensor],
) -> t.Union[np.ndarray, torch.Tensor]:
    """
    :param points: tensor of shape (N, 3) representing floating point coordinates
    :param transformation: (4, 4) tensor of a transformation matrix
    :return: transformed points
    """
    if all(isinstance(i, np.ndarray) for i in [points, transformation]):
        transformed_points = np.matmul(
            transformation,
            np.concatenate(
                [points[:, 0:3], np.ones(shape=(points.shape[0], 1))], axis=-1
            ).T,
        ).T
    elif all(isinstance(i, torch.Tensor) for i in [points, transformation]):
        transformed_points = torch.matmul(
            transformation,
            torch.concat(
                [points[:, 0:3], torch.ones(size=(points.shape[0], 1))], dim=-1
            ).T,
        ).T
    else:
        raise TypeError("Both inputs should be either np.ndarray or torch.Tensor type.")
    points[:, 0:3] = transformed_points[:, 0:3]
    return points


def find_corr(xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=NN_MAX_N)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def demo_alignment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    voxel_size = config.voxel_size
    checkpoint = torch.load(config.model)

    # init model
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)

    input_pcd = o3d.io.read_point_cloud(config.input)
    # create fixed input (points and features)
    fixed_xyz = np.array(input_pcd.points)
    fixed_feats = np.ones((len(fixed_xyz), 1))

    # create moving input (points and features)
    moving_xyz = np.array(input_pcd.points)
    moving_feats = np.ones((len(fixed_feats), 1))

    # randomly transform moving input
    random_transform = np.array(
        [[0.57620317, 0.68342775, -0.44823694, 2.4],
         [0.20656687, -0.65240175, -0.729179, 1.0],
         [-0.7907718, 0.3275644, -0.5170895, 3.9],
         [0., 0., 0., 1.]]
    )
    # draw_tensor_points([moving_xyz, apply_transformation(moving_xyz.copy(), random_transform)])
    moving_xyz = apply_transformation(moving_xyz, random_transform)

    # create fixed sparse tensor and model features
    # voxelize xyz and feats
    fixed_coords = np.floor(fixed_xyz / voxel_size)
    fixed_coords, fixed_inds = ME.utils.sparse_quantize(fixed_coords, return_index=True)
    # convert to batched coords compatible with ME
    fixed_coords = ME.utils.batched_coordinates([fixed_coords])
    fixed_unique_xyz = fixed_xyz[fixed_inds]
    fixed_feats = fixed_feats[fixed_inds]
    fixed_tensor = ME.SparseTensor(
        torch.tensor(fixed_feats, dtype=torch.float32),
        coordinates=torch.tensor(fixed_coords, dtype=torch.int32),
        device=device
    )

    # create moving sparse tensor and model features
    moving_coords = np.floor(moving_xyz / voxel_size)
    moving_coords, moving_inds = ME.utils.sparse_quantize(moving_coords, return_index=True)
    # convert to batched coords compatible with ME
    moving_coords = ME.utils.batched_coordinates([moving_coords])
    moving_unique_xyz = moving_xyz[moving_inds]
    moving_feats = moving_feats[moving_inds]
    moving_tensor = ME.SparseTensor(
        torch.tensor(moving_feats, dtype=torch.float32),
        coordinates=torch.tensor(moving_coords, dtype=torch.int32),
        device=device
    )

    # visualize inputs to be aligned
    draw_tensor_points([fixed_unique_xyz, moving_unique_xyz])

    # get model features of inputs
    fixed_model_feats = model(fixed_tensor).F
    moving_model_feats = model(moving_tensor).F

    # visualize model features
    # fixed_vis_pcd = o3d.geometry.PointCloud()
    # fixed_vis_pcd.points = o3d.utility.Vector3dVector(fixed_unique_xyz)
    # fixed_vis_pcd = get_colored_point_cloud_feature(
    #     fixed_vis_pcd,
    #     fixed_model_feats.detach().cpu().numpy(),
    #     voxel_size)
    # o3d.visualization.draw_geometries([fixed_vis_pcd])
    #
    # moving_vis_pcd = o3d.geometry.PointCloud()
    # moving_vis_pcd.points = o3d.utility.Vector3dVector(moving_unique_xyz)
    # moving_vis_pcd = get_colored_point_cloud_feature(
    #     moving_vis_pcd,
    #     moving_model_feats.detach().cpu().numpy(),
    #     voxel_size)
    # o3d.visualization.draw_geometries([moving_vis_pcd])

    # compute correspondences and alignment
    xyz0_corr, xyz1_corr = find_corr(
        torch.tensor(moving_unique_xyz, dtype=torch.float32).to(device),
        torch.tensor(fixed_unique_xyz, dtype=torch.float32).to(device),
        moving_model_feats,
        fixed_model_feats,
        subsample_size=SUBSAMPLE_SIZE,
    )
    xyz0_corr, xyz1_corr = xyz0_corr.cpu(), xyz1_corr.cpu()

    # estimate transformation using the correspondences
    est_transformation = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

    # transform moving points
    aligned_xyz = apply_transformation(moving_xyz.copy(), est_transformation.numpy())

    # visualize the results
    draw_tensor_points([fixed_xyz, aligned_xyz])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default='redkitchen-20.ply',
        type=str,
        help='path to a pointcloud file')
    parser.add_argument(
        '-m',
        '--model',
        default='ResUNetBN2C-16feat-3conv.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')

    config = parser.parse_args()
    demo_alignment(config)
