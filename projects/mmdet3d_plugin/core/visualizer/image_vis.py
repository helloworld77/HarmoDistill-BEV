# Copyright (c) OpenMMLab. All rights reserved.
import copy

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
    """Project the 3D points cloud on 2D image.

    Args:
        points (numpy.array): 3D points cloud (x, y, z) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        max_distance (float, optional): the max distance of the points cloud.
            Default: 70.
        thickness (int, optional): The thickness of 2D points. Default: -1.
    """
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=1,
            color=tuple(color),
            thickness=thickness,
        )
    cv2.imshow('project_pts_img', img.astype(np.uint8))
    cv2.waitKey(100)


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1, 
                       bboxes2d=None):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            if abs(corners[start, 0]) >= 5*img.shape[1] or abs(corners[start, 1]) >= 5*img.shape[0] or abs(corners[end, 0]) >= 5*img.shape[1] or abs(corners[end, 1]) >= 5*img.shape[0]:
                continue # streampetr not distinguish boxes from different picture
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)
    if bboxes2d is not None:
        for box in bboxes2d:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3]),), (255, 0, 255), thickness)
    return img.astype(np.uint8)

def box3d_to_corners(box3d):
    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
    CNS, YNS = 0, 1  # centerness and yawness indices in qulity
    YAW = 6  # decoded

    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners

def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             color=(0, 255, 0),
                             thickness=1,
                             bboxes2d=None, 
                             center_points=None,
                             point_weight=None):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    # corners_3d = bboxes3d.corners
    if bboxes3d is not None:
        corners_3d = box3d_to_corners(bboxes3d)
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
            np.ones((num_bbox * 8, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
        img = plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness, bboxes2d=bboxes2d)
    if center_points is not None:
        img = draw_points_on_img(center_points, img, lidar2img_rt, color=(0, 0, 255), circle=4, point_weight=point_weight)
    return img


# TODO: remove third parameter in all functions here in favour of img_metas
def draw_depth_bbox3d_on_img(bboxes3d,
                             raw_img,
                             calibs,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
            3d bbox in depth coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        calibs (dict): Camera calibration information, Rt and K.
        img_metas (dict): Used in coordinates transformation.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img
    from mmdet3d.models import apply_3d_transformation

    img = raw_img.copy()
    img_metas = copy.deepcopy(img_metas)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # first reverse the data transformations
    xyz_depth = apply_3d_transformation(
        points_3d, 'DEPTH', img_metas, reverse=True)

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(xyz_depth,
                               xyz_depth.new_tensor(img_metas['depth2img']))
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_camera_bbox3d_on_img(bboxes3d,
                              raw_img,
                              cam2img,
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4, point_weight=None):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            if point[0] < 0 or point[0] >= img.shape[1] or point[1] < 0 or point[1] >= img.shape[0]:
                continue
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), background=200, thickness=2, points=None):
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.ones([bev_h, bev_w, 3])*background

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),
            int((cir + 1) * 10 / bev_resolution),
            marking_color,
            thickness=thickness,
        )
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
    )
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
    )
    if bboxes_3d is not None and len(bboxes_3d) != 0:
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][
            ..., [0, 1]
        ]
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                )
    if points is not None:
        for point in points[0]:
            x = point[0] / bev_resolution + bev_w / 2
            y = -point[1] / bev_resolution + bev_h / 2
            cv2.circle(bev, (int(x), int(y)), 2, (0, 0, 255), thickness=-1)
            
    return bev.astype(np.uint8)

def draw_input_img(img, bs_id=0, img_norm_mean=[123.675, 116.28, 103.53], img_norm_std=[58.395, 57.12, 57.375]):
    if img.shape[0] // 6 > 1:
        img = img[6*bs_id: 6*(bs_id+1)]
    raw_imgs = img.cpu().numpy().transpose(0, 2, 3, 1)
    raw_imgs = raw_imgs * img_norm_std + img_norm_mean
    vis_imgs = []
    for i in range(raw_imgs.shape[0]):
        vis_imgs.append(raw_imgs[i].astype(np.uint8))
    vis_imgs = np.concatenate([
        np.concatenate(vis_imgs[:raw_imgs.shape[0]//2], axis=1),
        np.concatenate(vis_imgs[raw_imgs.shape[0]//2:], axis=1)
    ], axis=0)
    plt.imshow(vis_imgs)
    plt.show()

def draw_bbox3d_on_img(data, targets, img_metas, bs_id=0, center_points=None, show=False):
    raw_imgs = data["img"][bs_id].permute(0, 2, 3, 1).cpu().numpy()
    img_norm_mean = img_metas[bs_id]['img_norm_cfg']['mean']
    img_norm_std = img_metas[bs_id]['img_norm_cfg']['std']
    raw_imgs = raw_imgs * img_norm_std + img_norm_mean
    vis_imgs = []
    ## draw 3d bbox
    for i, (img, lidar2img) in enumerate(zip(raw_imgs, data['lidar2img'][bs_id])):
        img = img.astype(np.uint8)
        # vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id], img, lidar2img, (255, 0, 0), bboxes2d=data['gt_bboxes'][bs_id][i], center_points=targets[bs_id][:,:3]))
        # vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id], img, lidar2img, (255, 0, 0), bboxes2d=data['gt_bboxes'][bs_id][i]))
        if targets is None:
            vis_imgs.append(draw_lidar_bbox3d_on_img(None, img, lidar2img, (255, 0, 0), bboxes2d=None, center_points=center_points))
        elif type(targets[bs_id]) == torch.Tensor:
            vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id], img, lidar2img, (255, 0, 0), bboxes2d=None, center_points=center_points))
        else:
            vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id].tensor, img, lidar2img, (255, 0, 0), bboxes2d=None, center_points=center_points))
    vis_imgs = np.concatenate([
        np.concatenate(vis_imgs[:raw_imgs.shape[0]//2], axis=1),
        np.concatenate(vis_imgs[raw_imgs.shape[0]//2:], axis=1)
    ], axis=0)
    ### draw bev
    if targets is None:
        vis_bev = draw_lidar_bbox3d_on_bev(None, vis_imgs.shape[0], color=(255, 0, 0), points=center_points)
    elif type(targets[bs_id]) == torch.Tensor:
        vis_bev = draw_lidar_bbox3d_on_bev(targets[bs_id], vis_imgs.shape[0], color=(255, 0, 0))
    else:
        vis_bev = draw_lidar_bbox3d_on_bev(targets[bs_id].tensor, vis_imgs.shape[0], color=(255, 0, 0), points=center_points)
    vis_imgs = np.concatenate([vis_bev, vis_imgs], axis=1)
    if show:
        plt.figure(figsize=(20, 10))
        plt.imshow(vis_imgs)
        # 保存图像到文件
        plt.savefig('saved_image%s.png'%img_metas[bs_id]['scene_token'])
    else:
        return vis_imgs

def draw_bbox3d_on_img_3x2(data, targets, img_metas, bs_id=0, center_points=None, show=False, color_pv=(255, 0, 0), color_bev=(0, 255, 0), background=200, thickness=2, bev_range=115):
    raw_imgs = data["img"][bs_id].permute(0, 2, 3, 1).cpu().numpy()
    img_norm_mean = img_metas[bs_id]['img_norm_cfg']['mean']
    img_norm_std = img_metas[bs_id]['img_norm_cfg']['std']
    raw_imgs = raw_imgs * img_norm_std + img_norm_mean
    vis_imgs = []
    # 0前，1右前，2左前, 3后，4左后，5右后
    ## draw 3d bbox
    for i, (img, lidar2img) in enumerate(zip(raw_imgs, data['lidar2img'][bs_id])):
        img = img.astype(np.uint8)
        # vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id], img, lidar2img, (255, 0, 0), bboxes2d=data['gt_bboxes'][bs_id][i], center_points=targets[bs_id][:,:3]))
        # vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id], img, lidar2img, (255, 0, 0), bboxes2d=data['gt_bboxes'][bs_id][i]))
        if type(targets[bs_id]) == torch.Tensor:
            vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id], img, lidar2img, color=(255, 0, 0), bboxes2d=None))
        else:
            vis_imgs.append(draw_lidar_bbox3d_on_img(targets[bs_id].tensor, img, lidar2img, color=color_pv, bboxes2d=None))
    ## 图像左右翻转
    vis_imgs[0] = cv2.flip(vis_imgs[0], 1)
    vis_imgs_left = np.concatenate([cv2.flip(vis_imgs[1], 1), cv2.flip(vis_imgs[0], 1), cv2.flip(vis_imgs[2], 1),], axis=0)
    vis_imgs_right = np.concatenate([cv2.flip(vis_imgs[5], 1), cv2.flip(vis_imgs[3], 1), cv2.flip(vis_imgs[4], 1),], axis=0)
    ### draw bev
    if type(targets[bs_id]) == torch.Tensor:
        vis_bev = draw_lidar_bbox3d_on_bev(targets[bs_id], vis_imgs_left.shape[0], color=(255, 0, 0))
    else:
        vis_bev = draw_lidar_bbox3d_on_bev(targets[bs_id].tensor, vis_imgs_left.shape[0], color=color_bev, background=background, thickness=thickness, bev_range=bev_range)
        ## 左旋90
        vis_bev = cv2.rotate(vis_bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _vis = np.concatenate([vis_imgs_left, vis_bev, vis_imgs_right], axis=1)
    if show:
        plt.figure(figsize=(20, 10))
        plt.imshow(_vis)
        # 保存图像到文件
        plt.savefig('saved_image%s.png'%img_metas[bs_id]['scene_token'])
    else:
        return _vis
    
def draw_points_on_img_v2(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    pts_4d = np.concatenate([points.cpu().numpy(), np.ones((points.shape[0], 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    img = img.copy()
    for point in pts_2d[:, :2]:
        if point[0] < 0 or point[0] >= img.shape[1] or point[1] < 0 or point[1] >= img.shape[0]:
            continue
        cv2.circle(img, point.astype(np.int), circle, color, thickness=3)
    return img.astype(np.uint8)

def plot_raydn_known_bbox_center(data, known_bbox_center, img_metas, bs_id, color=(255, 0, 0), show=False):
    raw_imgs = data["img"][bs_id].permute(0, 2, 3, 1).cpu().numpy()
    raw_imgs = raw_imgs * img_metas[bs_id]['img_norm_cfg']['std'] + img_metas[bs_id]['img_norm_cfg']['mean']
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(raw_imgs, data['lidar2img'][bs_id])):
        img = img.astype(np.uint8)
        vis_imgs.append(draw_points_on_img_v2(known_bbox_center, img, lidar2img, color))
    vis_imgs = np.concatenate([
        np.concatenate(vis_imgs[:raw_imgs.shape[0]//2], axis=1),
        np.concatenate(vis_imgs[raw_imgs.shape[0]//2:], axis=1)
    ], axis=0)
    if show:
        plt.figure(figsize=(20, 10))
        plt.imshow(vis_imgs)
        plt.savefig('known_bbox_center%s.png'%img_metas[bs_id]['scene_token'])
    else:
        return vis_imgs