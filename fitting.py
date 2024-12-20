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
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils
from smplx.lbs import transform_mat

# import mesh_to_sdf
from pysdf import SDF
import trimesh

# https://github.com/zju3dv/instant-nvr/blob/master/lib/utils/blend_utils.py
NUM_PARTS = 5
part_bw_map = {
    'body': [14, 13, 9, 6, 3, 0],
    'leg': [1, 2, 4, 5, 7, 8, 10, 11],
    'head': [12, 15],
    'larm': [16, 18, 20, 22],
    'rarm': [17, 19, 21, 23],
}
partnames = ['body', 'leg', 'head', 'larm', 'rarm']


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl', max_persons=2,
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type
        self.max_persons = max_persons

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mvs = [MeshViewer(body_color=self.body_color) for _ in range(self.max_persons)]
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            for person_id in range(self.max_persons):
                self.mvs[person_id].close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_models,
                    use_vposer=True, pose_embeddings=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                for person_id in range(self.max_persons):
                    body_pose = vposer.decode(
                        pose_embeddings[person_id], output_type='aa').view(
                            1, -1) if use_vposer else None

                    if append_wrists:
                        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                dtype=body_pose.dtype,
                                                device=body_pose.device)
                        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                    model_output = body_models[person_id](
                        return_verts=True, body_pose=body_pose)
                    vertices = model_output.vertices.detach().cpu().numpy()

                    self.mvs[person_id].update_mesh(vertices.squeeze(),
                                        body_models[person_id].faces)

            prev_loss = loss.item()

        return prev_loss


    def create_fitting_closure_multiview(self,
                                         optimizer, body_models,
                                         camera_list=None, global_body_translations=None,
                                         body_model_scale=None,
                                         gt_joints_list=None, loss_list=None,
                                         joints_conf_list=None,
                                         joint_weights=None,
                                         return_verts=True, return_full_pose=False,
                                         use_vposer=False, vposer=None,
                                         pose_embeddings=None,
                                         create_graph=False,
                                         inter_person_loss_list=None,
                                         mask_list=None,
                                         **kwargs):
        faces_tensors = [body_models[person_id].faces_tensor.view(-1) for person_id in range(len(body_models))]
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            total_loss = 0
            body_model_outputs = []
            body_poses = []
            for person_id in range(len(body_models)):
                body_pose = vposer.decode(
                    pose_embeddings[person_id], output_type='aa').view(
                    1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                            dtype=body_pose.dtype,
                                            device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                body_model_output = body_models[person_id](return_verts=return_verts,
                                            body_pose=body_pose,
                                            return_full_pose=return_full_pose)
                body_model_outputs.append(body_model_output)
                body_poses.append(body_pose)

                losses = []
                for i in range(len(camera_list)):
                    loss = loss_list[i][person_id]
                    if loss is None:
                        continue
                    l = loss(body_model_output, camera=camera_list[i],
                                    global_body_translation=global_body_translations[person_id],
                                    body_model_scale=body_model_scale,
                                    gt_joints=gt_joints_list[i][person_id],
                                    body_model_faces=faces_tensors[person_id],
                                    joints_conf=joints_conf_list[i][person_id],
                                    joint_weights=joint_weights,
                                    pose_embedding=pose_embeddings[person_id],
                                    use_vposer=use_vposer,
                                    **kwargs)
                    losses.append(l)
                    total_loss += l
            inter_person_loss_total = 0.
            # for i in range(len(camera_list)):
            #     inter_person_loss = inter_person_loss_list[i]
            #     if inter_person_loss is None or (inter_person_loss.coll_loss_weight == 0).all():
            #         continue
            #     mask = torch.tensor(mask_list[i], device=body_poses[0].device)
            #     inter_person_loss_total += inter_person_loss(body_models, body_poses, camera_list[i], 
            #                                     global_body_translations, body_model_scale, faces_tensors[0], mask)
            total_loss = total_loss + inter_person_loss_total # * 0.115
            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                for person_id in range(len(body_models)):
                    model_output = body_models[person_id](return_verts=True,
                                            body_pose=body_pose)
                    vertices = model_output.vertices.detach().cpu().numpy()

                    self.mvs[person_id].update_mesh(vertices.squeeze(),
                                        body_models[person_id].faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 body_model_scale=1.0,
                 reduction='sum',
                 prev_verts=None,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))
        
        self.prev_verts = prev_verts

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, global_body_translation,
                body_model_scale,
                gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):
        projected_joints = camera(
            body_model_scale * body_model_output.joints + global_body_translation)
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))
        if self.prev_verts is not None:
            temporal_loss = torch.nn.functional.mse_loss(body_model_output.vertices[0], self.prev_verts)
        else:
            temporal_loss = 0.0
        # print(f"JOINT: {joint_loss.item():11.5f} PPRIOR: {pprior_loss.item():11.5f} SHAPE: {shape_loss.item():11.5f} ANGLE: {angle_prior_loss.item():11.5f} PEN: {pen_loss:11.5f}")
        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss + temporal_loss)
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss


class CollisionLoss(nn.Module):

    def __init__(self, dtype=torch.float32, coll_loss_weight=0.0,):

        super(CollisionLoss, self).__init__()

        # self.search_tree = search_tree
        # self.tri_filtering_module = tri_filtering_module
        # self.pen_distance = pen_distance

        self.register_buffer('coll_loss_weight',
                            torch.tensor(coll_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_models, body_poses, camera, global_body_translations,
                body_model_scale, body_model_face, mask):
        
        body_model_outputs = []
        for person_id in range(len(body_models)):
            betas = body_models[person_id].betas
            body_model_output = body_models[person_id](return_verts=True,
                                                body_pose=body_poses[person_id],
                                                return_full_pose=False,
                                                betas=betas.detach())
            body_model_outputs.append(body_model_output)
        assert len(body_model_outputs) == 2

        body_model_face = body_model_face.reshape(-1, 3).cpu().detach().numpy()
        intruder_indices = [0, 1]
        receiver_indices = [1, 0]
        verts_colls = []
        sdf_masks = []
        for i in range(2):
            int_idx = intruder_indices[i]
            rec_idx = receiver_indices[i]
            # receiver = trimesh.Trimesh(vertices=body_model_outputs[rec_idx].vertices.cpu().detach().numpy()[0], faces=body_model_face.cpu().detach().numpy())
            rec_verts = body_model_outputs[rec_idx].vertices[0] + global_body_translations[rec_idx][None]
            int_verts = body_model_outputs[int_idx].vertices[0] + global_body_translations[int_idx][None]
            # sdf = mesh_to_sdf.mesh_to_sdf(receiver, int_verts.cpu().detach().numpy(), surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
            sdf_f = SDF(rec_verts.cpu().detach().numpy(), body_model_face) # receiver.vertices, receiver.faces)
            sdf = sdf_f(int_verts.cpu().detach().numpy())
            sdf_mask = sdf > 0 # < 0
            verts_coll = int_verts[sdf_mask, :]
            verts_colls.append(verts_coll)
            sdf_masks.append(sdf_mask)

        m = 1.0
        ranking_loss = 0.0
        for i in range(2):
            int_idx = intruder_indices[i]
            rec_idx = receiver_indices[i]
            rec_verts = verts_colls[rec_idx]
            int_verts = verts_colls[int_idx]
            
            int_part_id = body_models[person_id].lbs_weights.argmax(dim=-1)
            int_coll_id = int_part_id[sdf_masks[int_idx]]
            for id in int_coll_id.unique():
                pull_verts = int_verts[int_coll_id == id]
                mean_pos = pull_verts.mean(dim=0)
                threshold = torch.amax(torch.norm(pull_verts - mean_pos, dim=-1))
                rec_dist = torch.norm(rec_verts - mean_pos, dim=-1)
                push_verts = rec_verts[rec_dist < threshold]
                if len(push_verts) == 0:
                    continue
                pos_dist = torch.norm(pull_verts - mean_pos, dim=-1).mean()
                neg_dist = torch.norm(push_verts - mean_pos, dim=-1).mean()
                loss = torch.clamp(m + pos_dist - neg_dist, min=0.0)
                ranking_loss = ranking_loss + loss
        
        return ranking_loss