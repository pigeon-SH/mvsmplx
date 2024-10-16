import json
import argparse

# body_map = {
#     # coco: openpose
#     0: 0,  # Nose
#     1: 16, # Left Eye
#     2: 15, # Right Eye
#     3: 18, # Left Ear
#     4: 17, # Right Ear
#     5: 5,  # Left Shoulder
#     6: 2,  # Right Shoulder
#     7: 6,  # Left Elbow
#     8: 3,  # Right Elbow
#     9: 7,  # Left Wrist
#     10: 4, # Right Wrist
#     11: 12,# Left Hip
#     12: 9, # Right Hip
#     13: 13,# Left Knee
#     14: 10, # Right Knee
#     15: 14,# Left Ankle
#     16: 11, # Right Ankle,
# 	17: 19, # Lbigtoe
# 	18: 20, # Lsmalltoe
# 	19: 21, # Lheel
# 	20: 22, # Rbigtoe
# 	21: 23, # Rsmalltoe
# 	22: 24, # Rheel
#     -1: 1,  # Neck
# }
body_map = {
    # coco: openpose
    0: 0,  # Nose
    1: -1,  # Neck
    2: 6,  # Right Shoulder
    3: 8,  # Right Elbow
    4: 10,  # Right Wrist
    5: 5,  # Left Shoulder
    6: 7,  # Left Elbow
    7: 9,  # Left Wrist
    8: -1,  # Pelvis
    9: 12,  # Right Hip
    10: 14,  # Right Knee
    11: 16,  # Right Ankle,
    12: 11,  # Left Hip
    13: 13,  # Left Knee
    14: 15,  # Left Ankle
    15: 2,  # Right Eye
    16: 1,  # Left Eye
    17: 4,  # Right Ear
    18: 3,  # Left Ear
    19: 17,  # Lbigtoe
    20: 18,  # Lsmalltoe
    21: 19,  # Lheel
    22: 20,  # Rbigtoe
    23: 21,  # Rsmalltoe
    24: 22,  # Rheel
}


def convert(openpose_path, coco_path, ordered=False):
    NUM_BODY_OPENPOSE = 25
    NUM_FACE_OPENPOSE = 70
    NUM_HAND_OPENPOSE = 21  # per hand
    NUM_BODY_COCO = 17
    NUM_FOOT_COCO = 3  # per foot
    NUM_FACE_COCO = 68
    NUM_HAND_COCO = 21  # per hand

    NUM_PERSON = 2
    openpose = {"people": []}
    with open(coco_path) as f:
        coco = json.load(f)["instance_info"]
    for person_id in range(len(coco)):
        person_info = {
            "person_id": person_id,
            "pose_keypoints_2d": [],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }
        # body
        for i in range(NUM_BODY_OPENPOSE):
            coco_idx = body_map[i]
            if coco_idx < 0:
                kpt = [0.0, 0.0]
                score = 0.0
            else:
                kpt = coco[person_id]["keypoints"][coco_idx]
                score = coco[person_id]["keypoint_scores"][coco_idx]
            person_info["pose_keypoints_2d"] += [kpt[0], kpt[1], score]
        assert len(person_info["pose_keypoints_2d"]) == NUM_BODY_OPENPOSE * 3

        # face
        start_idx = NUM_BODY_COCO + NUM_FOOT_COCO * 2
        end_idx = start_idx + NUM_FACE_COCO
        kpts = coco[person_id]["keypoints"][start_idx:end_idx]
        scores = coco[person_id]["keypoint_scores"][start_idx:end_idx]
        for i in range(NUM_FACE_COCO):
            kpt = kpts[i]
            score = scores[i]
            person_info["face_keypoints_2d"] += [kpt[0], kpt[1], score]
        person_info["face_keypoints_2d"] += [0.0, 0.0, 0.0]  # pupil
        person_info["face_keypoints_2d"] += [0.0, 0.0, 0.0]  # pupil
        assert len(person_info["face_keypoints_2d"]) == NUM_FACE_OPENPOSE * 3

        # left hand
        start_idx = NUM_BODY_COCO + NUM_FOOT_COCO * 2 + NUM_FACE_COCO
        end_idx = start_idx + NUM_HAND_COCO
        kpts = coco[person_id]["keypoints"][start_idx:end_idx]
        scores = coco[person_id]["keypoint_scores"][start_idx:end_idx]
        for i in range(NUM_HAND_COCO):
            kpt = kpts[i]
            score = scores[i]
            person_info["hand_left_keypoints_2d"] += [kpt[0], kpt[1], score]
        assert len(person_info["hand_left_keypoints_2d"]) == NUM_HAND_OPENPOSE * 3

        # right hand
        start_idx = NUM_BODY_COCO + NUM_FOOT_COCO * 2 + NUM_FACE_COCO + NUM_HAND_COCO
        end_idx = start_idx + NUM_HAND_COCO
        kpts = coco[person_id]["keypoints"][start_idx:end_idx]
        scores = coco[person_id]["keypoint_scores"][start_idx:end_idx]
        for i in range(NUM_HAND_COCO):
            kpt = kpts[i]
            score = scores[i]
            person_info["hand_right_keypoints_2d"] += [kpt[0], kpt[1], score]
        assert len(person_info["hand_right_keypoints_2d"]) == NUM_HAND_OPENPOSE * 3

        openpose["people"].append(person_info)

    with open(openpose_path, "w") as f:
        json.dump(openpose, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--openpose_path", type=str)
    args = parser.parse_args()
    convert(args.openpose_path, args.coco_path)
