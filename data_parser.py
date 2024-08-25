# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False, max_persons=2):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    keypoints_np = None
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

        if idx == 0:
            keypoints_np = np.zeros_like(body_keypoints)[None].repeat(repeats=max_persons, axis=0)
        keypoints_np[person_data['person_id']] = body_keypoints

    return Keypoints(keypoints=keypoints_np, gender_pd=gender_pd,
                     gender_gt=gender_gt)


def generate_cam_Rt(center, direction, right, up):
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    center = center.reshape([-1])
    direction = direction.reshape([-1])
    right = right.reshape([-1])
    up = up.reshape([-1])

    rot_mat = np.eye(3)
    s = right
    s = normalize_vector(s)
    rot_mat[0, :] = s
    u = up
    u = normalize_vector(u)
    rot_mat[1, :] = -u
    rot_mat[2, :] = normalize_vector(direction)
    trans = -np.dot(rot_mat, center)
    return rot_mat, trans


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='color',
                 keyp_folder='keypoints',
                 cam_subpath='meta/cam_data.mat',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 mask_folder='mask',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)
        self.cam_fpath = osp.join(data_folder, cam_subpath)
        self.mask_folder = osp.join(data_folder, mask_folder)

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if (img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.'))]
        self.img_paths = sorted(self.img_paths)
        assert len(self.img_paths) == 8
        # self.img_paths = self.img_paths[::2]    # PIGEONSH: SPARSE VIEW
        self.cnt = 0

        self.cam_param = np.load("/home/vclab/dataset/Hi4D/talk/talk01/cameras/rgb_cameras.npz")
        self.max_persons = kwargs['max_persons']

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        # read images
        img = cv2.imread(img_path).astype(np.float32)[:, :, :] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        # read key points
        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour, max_persons=self.max_persons)
        
        mask_fn = osp.join(self.mask_folder, img_fn + '.png')
        mask = cv2.imread(mask_fn).astype(np.float32)
        mask = update_mask_with_keypoints(mask, keyp_tuple.keypoints)

        # if len(keyp_tuple.keypoints) < 1:
        #     return {}
        keypoints = keyp_tuple.keypoints

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 
                       'img': img,
                       'mask': mask}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        
        # read camera
        # cam_id = int(img_fn)
        # cam_data = sio.loadmat(self.cam_fpath)['cam'][0]
        # cam_param = cam_data[cam_id]
        # cam_R, cam_t = generate_cam_Rt(
        #     center=cam_param['center'][0, 0], right=cam_param['right'][0, 0],
        #     up=cam_param['up'][0, 0], direction=cam_param['direction'][0, 0])
        # cam_R = cam_R.astype(np.float32)
        # cam_t = cam_t.astype(np.float32)
        # cam_r = np.float32(cam_data['cam_rs'][cam_id])
        # cam_t = np.float32(cam_data['cam_ts'][cam_id])
        # cam_R = cv2.Rodrigues(cam_r)[0]
        ##### PIGEONSH
        cam_id = int(img_path.split('.')[-2].split('_')[-1])
        cam_idx = np.arange(len(self.cam_param['ids']))[self.cam_param['ids'] == cam_id].item()
        K = self.cam_param['intrinsics'][cam_idx]
        RT = self.cam_param['extrinsics'][cam_idx]
        # self.intrins = torch.tensor([K[0, 0], K[1, 1], self.img_size[0]//2, self.img_size[1]//2], dtype=torch.float32)[None].repeat(self.seq_len, 1)
        # self.cam_R = torch.tensor(RT[:3, :3], dtype=torch.float32)[None].repeat(self.seq_len, 1, 1)
        # self.cam_T = torch.tensor(RT[:3, 3], dtype=torch.float32)[None].repeat(self.seq_len, 1)
        # self.is_static=True
        # img_w = K[0, 2] * 2
        # img_h = K[1, 2] * 2

        output_dict['cam_id'] = cam_id
        output_dict['cam_R'] = np.float32(RT[:3, :3])
        output_dict['cam_t'] = np.float32(RT[:3, 3])
        output_dict['cam_fx'] = K[0, 0]
        output_dict['cam_fy'] = K[1, 1]
        output_dict['cam_cx'] = K[0, 2] # img.shape[1] / 2
        output_dict['cam_cy'] = K[1, 2] # img.shape[0] / 2

        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)


def update_mask_with_keypoints(mask, keypoints):
    from scipy.optimize import linear_sum_assignment
    # 고유한 mask 값 추출 (배경인 0 제외)
    unique_values = np.unique(mask.reshape(-1, 3), axis=0)
    unique_values = unique_values[1:]
    
    num_values = len(unique_values)
    num_people = keypoints.shape[0]
    
    # 겹침 정도를 저장할 매트릭스 초기화 (values x people)
    overlap_matrix = np.zeros((num_values, num_people), dtype=int)
    
    # 각 고유 값에 대해 계산
    for i, value in enumerate(unique_values):
        # 현재 value의 영역 마스크
        value_mask = (mask == value).sum(axis=-1)
        
        # 각 사람에 대해
        for j in range(num_people):
            person_overlap = 0
            
            # 해당 사람의 keypoint 좌표들 가져오기
            person_keypoints = keypoints[j, :, :2].astype(int)
            
            # keypoint가 value 영역과 얼마나 겹치는지 계산
            for x, y in person_keypoints:
                if value_mask[y, x]:
                    person_overlap += 1
            
            # 겹치는 정도를 매트릭스에 저장
            overlap_matrix[i, j] = person_overlap
    
    # 겹침 정도를 최대화하는 매칭 찾기 (헝가리안 알고리즘 사용)
    row_indices, col_indices = linear_sum_assignment(-overlap_matrix)
    
    # 새로운 mask를 기존 mask의 복사본으로 생성
    new_mask = np.copy(mask)
    
    # 매칭된 결과를 이용하여 mask 값 업데이트
    for row_idx, col_idx in zip(row_indices, col_indices):
        value = unique_values[row_idx]
        person_idx = col_idx + 1  # 사람 인덱스는 1부터 시작하므로 +1
        new_mask[mask == value] = person_idx
        if person_idx == 2:
            assert value[0] == 125
        else:
            assert value[0] == 255
    
    return new_mask