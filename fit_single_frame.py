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


import time
import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from vposer.model_loader import load_vposer


def fit_single_frame(img_list,
                     keypoints_list,
                     body_models,
                     camera_list,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     output_folder='',
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    assert len(img_list) == len(keypoints_list)
    assert len(img_list) == len(camera_list)

    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    max_persons = kwargs['max_persons']

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embeddings = [None, ] * 2
    if use_vposer:
        pose_embeddings = torch.zeros([max_persons, 32],
                                    dtype=dtype, device=device,
                                    requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_poses = torch.zeros([max_persons, vposer_latent_dim],
                                     dtype=dtype)
    else:
        breakpoint()
        body_mean_poses = body_pose_prior.get_mean().detach().cpu()

    view_num = len(camera_list)
    loss_list = list()
    gt_joints_list = list()
    joints_conf_list = list()

    assert(view_num > 0)
    inter_person_loss_list = []
    for view_id in range(view_num):
        valid_person_cnt = 0
        loss_list_person = list()
        gt_joints_list_person = list()
        joints_conf_list_person = list()
        for person_id in range(max_persons):
            if keypoints_list[view_id][person_id].sum() == 0:
                loss_list_person.append(None)
                gt_joints_list_person.append(None)
                joints_conf_list_person.append(None)
                continue
            keypoint_data = torch.tensor(keypoints_list[view_id][person_id:person_id+1], dtype=dtype)
            valid_person_cnt += 1
            gt_joints = keypoint_data[:, :, :2]
            if use_joints_conf:
                joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

            # Transfer the data to the correct device
            gt_joints = gt_joints.to(device=device, dtype=dtype)
            gt_joints_list_person.append(gt_joints)
            if use_joints_conf:
                joints_conf = joints_conf.to(device=device, dtype=dtype)
                joints_conf_list_person.append(joints_conf)

            # Create the search tree
            search_tree = None
            pen_distance = None
            filter_faces = None
            if interpenetration:
                # raise NotImplementedError('The interpenetration constraint was removed!')
                from mesh_intersection.bvh_search_tree import BVH
                import mesh_intersection.loss as collisions_loss
                from mesh_intersection.filter_faces import FilterFaces

                assert use_cuda, 'Interpenetration term can only be used with CUDA'
                assert torch.cuda.is_available(), \
                    'No CUDA Device! Interpenetration term can only be used' + \
                    ' with CUDA'

                search_tree = BVH(max_collisions=max_collisions)

                pen_distance = \
                    collisions_loss.DistanceFieldPenetrationLoss(
                        sigma=df_cone_height, point2plane=point2plane,
                        vectorized=True, penalize_outside=penalize_outside)

                if part_segm_fn:
                    # Read the part segmentation
                    part_segm_fn = os.path.expandvars(part_segm_fn)
                    with open(part_segm_fn, 'rb') as faces_parents_file:
                        face_segm_data = pickle.load(faces_parents_file,
                                                    encoding='latin1')
                    faces_segm = face_segm_data['segm']
                    faces_parents = face_segm_data['parents']
                    # Create the module used to filter invalid collision pairs
                    filter_faces = FilterFaces(
                        faces_segm=faces_segm, faces_parents=faces_parents,
                        ign_part_pairs=ign_part_pairs).to(device=device)

            fct = view_num

            # Weights used for the pose prior and the shape prior
            opt_weights_dict = {'data_weight': data_weights,
                                'body_pose_weight': body_pose_prior_weights,
                                'shape_weight': shape_weights}
            # adjust energy weight for multi-view setup
            for i in range(len(opt_weights_dict['body_pose_weight'])):
                opt_weights_dict['body_pose_weight'][i] *= (view_num / fct)
            for i in range(len(opt_weights_dict['shape_weight'])):
                opt_weights_dict['shape_weight'][i] *= (view_num / fct)

            if use_face:
                opt_weights_dict['face_weight'] = face_joints_weights
                opt_weights_dict['expr_prior_weight'] = expr_weights
                opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
                for i in range(len(opt_weights_dict['expr_prior_weight'])):
                    opt_weights_dict['expr_prior_weight'][i] *= (view_num / fct)
                for i in range(len(opt_weights_dict['jaw_prior_weight'])):
                    opt_weights_dict['jaw_prior_weight'][i] *= (view_num / fct)

            if use_hands:
                opt_weights_dict['hand_weight'] = hand_joints_weights
                opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
                for i in range(len(opt_weights_dict['hand_prior_weight'])):
                    opt_weights_dict['hand_prior_weight'][i] *= (view_num / fct)

            if interpenetration:
                opt_weights_dict['coll_loss_weight'] = coll_loss_weights
                for i in range(len(opt_weights_dict['coll_loss_weight'])):
                    opt_weights_dict['coll_loss_weight'][i] *= (view_num / fct)

            keys = opt_weights_dict.keys()
            opt_weights = [dict(zip(keys, vals)) for vals in
                        zip(*(opt_weights_dict[k] for k in keys
                                if opt_weights_dict[k] is not None))]
            for weight_list in opt_weights:
                for key in weight_list:
                    weight_list[key] = torch.tensor(weight_list[key],
                                                    device=device,
                                                    dtype=dtype)

            loss = fitting.create_loss(loss_type=loss_type,
                                    joint_weights=joint_weights,
                                    rho=rho,
                                    use_joints_conf=use_joints_conf,
                                    use_face=use_face, use_hands=use_hands,
                                    vposer=vposer,
                                    pose_embedding=pose_embeddings[person_id],
                                    body_pose_prior=body_pose_prior,
                                    shape_prior=shape_prior,
                                    angle_prior=angle_prior,
                                    expr_prior=expr_prior,
                                    left_hand_prior=left_hand_prior,
                                    right_hand_prior=right_hand_prior,
                                    jaw_prior=jaw_prior,
                                    interpenetration=interpenetration,
                                    pen_distance=pen_distance,
                                    search_tree=search_tree,
                                    tri_filtering_module=filter_faces,
                                    # I scale the mesh model to [-0.5, 0.5] during rendering;
                                    # so I need to perform the same scaling
                                    # to make the body shapee plausible
                                    dtype=dtype,
                                    **kwargs)
            loss = loss.to(device=device)
            loss_list_person.append(loss)
        
        if valid_person_cnt > 1:
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            assert use_cuda, 'Interpenetration term can only be used with CUDA'
            assert torch.cuda.is_available(), \
                'No CUDA Device! Interpenetration term can only be used' + \
                ' with CUDA'

            search_tree = BVH(max_collisions=max_collisions)

            pen_distance = \
                collisions_loss.DistanceFieldPenetrationLoss(
                    sigma=df_cone_height, point2plane=point2plane,
                    vectorized=True, penalize_outside=penalize_outside)

            # PIGEONSH: no part_segm_fn

            inter_person_loss = fitting.InterPersonLoss(search_tree=search_tree, pen_distance=pen_distance)
        else:
            inter_person_loss = None
        loss_list.append(loss_list_person)
        gt_joints_list.append(gt_joints_list_person)
        joints_conf_list.append(joints_conf_list_person)
        inter_person_loss_list.append(inter_person_loss)

    body_scale = torch.tensor([1.0], dtype=dtype, device=device,)
                              # requires_grad=True)
    global_body_translations = torch.tensor([[0, 0, 0] for _ in range(max_persons)], dtype=dtype, device=device,
                                           requires_grad=True)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img_list[0], dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H

        # Reset the parameters to estimate the initial translation of the
        # body model
        for person_id in range(max_persons):
            body_models[person_id].reset_params(body_pose=body_mean_poses[person_id:person_id+1])

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        # shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
        #                            gt_joints[:, right_shoulder_idx])
        # try_both_orient = shoulder_dist.item() < side_view_thsh
        try_both_orient = False

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        orientations = []
        for person_id in range(max_persons):
            if try_both_orient:
                breakpoint()
                body_orient = body_models[person_id].global_orient.detach().cpu().numpy()
                flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                    cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
                flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

                flipped_orient = torch.tensor(flipped_orient,
                                            dtype=dtype,
                                            device=device).unsqueeze(dim=0)
                orientation = [body_orient, flipped_orient]
            else:
                orientation = [body_models[person_id].global_orient.detach().cpu().numpy()] 
            assert len(orientation) == 1 # PIGEONSH: try_both_orient is False
            orientations.append(orientation[0])

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        opt_start = time.time()

        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
            final_params = []
            for person_id in range(max_persons):
                new_params = defaultdict(global_orient=orientations[person_id],
                                    body_pose=body_mean_poses[person_id:person_id+1])
                body_models[person_id].reset_params(**new_params)
                if use_vposer:
                    with torch.no_grad():
                        pose_embeddings[person_id].fill_(0)

                body_params = list(body_models[person_id].parameters())

                final_params += list(
                    filter(lambda x: x.requires_grad, body_params))

            if use_vposer:
                final_params.append(pose_embeddings)
            final_params.append(global_body_translations)
            # final_params.append(body_scale)
            body_optimizer, body_create_graph = optim_factory.create_optimizer(
                final_params,
                **kwargs)
            body_optimizer.zero_grad()

            curr_weights['data_weight'] = data_weight
            curr_weights['bending_prior_weight'] = (
                3.17 * curr_weights['body_pose_weight'])
            if use_hands:
                joint_weights[:, 25:76] = curr_weights['hand_weight']
            if use_face:
                joint_weights[:, 76:] = curr_weights['face_weight']

            for i in range(len(loss_list)):
                for person_id in range(max_persons):
                    if loss_list[i][person_id] is None:
                        continue
                    loss_list[i][person_id].reset_loss_weights(curr_weights)
            for i in range(len(inter_person_loss_list)):
                    if inter_person_loss_list[i] is None:
                        continue
                    inter_person_loss_list[i].reset_loss_weights(curr_weights)

            closure = monitor.create_fitting_closure_multiview(
                body_optimizer, body_models,
                camera_list=camera_list, global_body_translations=global_body_translations,
                body_model_scale=body_scale,
                gt_joints_list=gt_joints_list,
                joints_conf_list=joints_conf_list,
                joint_weights=joint_weights,
                loss_list=loss_list, create_graph=body_create_graph,
                use_vposer=use_vposer, vposer=vposer,
                pose_embeddings=pose_embeddings,
                return_verts=True, return_full_pose=True,
                inter_person_loss_list=inter_person_loss_list)

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                stage_start = time.time()
            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                body_models,
                pose_embeddings=pose_embeddings, vposer=vposer,
                use_vposer=use_vposer)

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - stage_start
                if interactive:
                    tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                        opt_idx, elapsed))

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - opt_start
            tqdm.write('Body final loss val = {:.5f}'.format(
                final_loss_val))

        # Get the result of the fitting process
        # Store in it the errors list in order to compare multiple
        # orientations, if they exist
        # result = {'camera_' + str(key): val.detach().cpu().numpy()
        #           for key, val in camera.named_parameters()}
        for person_id in range(max_persons):
            result_person = {}
            result_person.update({key: val.detach().cpu().numpy()
                            for key, val in body_models[person_id].named_parameters()})
            result_person.update({'global_body_translation':
                                global_body_translations[person_id].detach().cpu().numpy()})
            if use_vposer:
                body_pose = vposer.decode(
                    pose_embeddings[person_id],
                    output_type='aa').view(1, -1) if use_vposer else None

                model_type = kwargs.get('model_type', 'smpl')
                append_wrists = model_type == 'smpl' and use_vposer
                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                dtype=body_pose.dtype,
                                                device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                result_person['body_pose'] = body_pose.detach().cpu().numpy()
                result_person['body_pose_embedding'] = pose_embeddings[person_id].detach().cpu().numpy()

            if result_person['body_pose'].shape[-1] == 69:
                body_pose = result_person['body_pose']
                body_pose = np.reshape(body_pose, (1, 69))
                body_pose = np.concatenate([result_person['global_orient'], body_pose], axis=1)
                result_person.update({'body_pose': body_pose})

            assert result_person['body_pose'].shape[-1] == 72
            results.append({'person_id': person_id,
                            'loss': final_loss_val,
                            'result': result_person})
        print('body_scale = %f' % body_scale.detach().cpu().numpy().squeeze())

        with open(result_fn, 'wb') as result_file:
            pickle.dump(results, result_file, protocol=2)

    if save_meshes or visualize:
        body_scale = body_scale.detach().cpu().numpy().squeeze()
        for person_id in range(max_persons):
            model_output = body_models[person_id](return_verts=True, body_pose=torch.from_numpy(results[person_id]['result']['body_pose'][:, 3:]).cuda())
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()

            # test projection
            global_trans = global_body_translations[person_id].detach().cpu().numpy().squeeze()

            # project smpl vertices onto images for debugging
            out_img_fd = osp.join(output_folder, 'vis')
            os.makedirs(out_img_fd, exist_ok=True)
            for i, (camera, img) in enumerate(zip(camera_list, img_list)):
                cam_fx = camera.focal_length_x.detach().cpu().numpy().squeeze()
                cam_fy = camera.focal_length_y.detach().cpu().numpy().squeeze()
                cam_c = camera.center.detach().cpu().numpy().squeeze()
                cam_trans = camera.translation.detach().cpu().numpy().squeeze()
                cam_rotation = camera.rotation.detach().cpu().numpy().squeeze()

                vertices_proj = vertices * body_scale + global_trans
                vertices_proj = np.dot(vertices_proj, cam_rotation.transpose())
                vertices_proj += np.expand_dims(cam_trans, axis=0)
                vertices_proj[:, 0] = vertices_proj[:, 0] * cam_fx / vertices_proj[:, 2] + cam_c[0]
                vertices_proj[:, 1] = vertices_proj[:, 1] * cam_fy / vertices_proj[:, 2] + cam_c[1]
                img_proj = np.copy(img)
                for v in vertices_proj:
                    v = np.int32(np.round(v))
                    v[0] = np.clip(v[0], 0, img_proj.shape[1]-1)
                    v[1] = np.clip(v[1], 0, img_proj.shape[0]-1)
                    img_proj[v[1], v[0], :] = np.asarray([0, 0, 1], dtype=np.float32)
                img_proj = np.uint8(img_proj*255)
                cv2.imwrite(osp.join(out_img_fd, '%04d_%d.png' % (i, person_id)), img_proj)

            import trimesh
            out_mesh = trimesh.Trimesh(vertices * body_scale + global_trans, body_models[person_id].faces)
            save_fn = mesh_fn.replace(".obj", f"_{person_id}.obj")
            out_mesh.export(save_fn)
