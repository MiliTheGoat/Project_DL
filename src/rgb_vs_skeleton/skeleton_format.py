"""The two skeleton formats of the project, and the shared 2D format used for cross-dataset testing.

"3d25"  the original NTU skeleton: 25 joints, x y z from the Kinect depth camera. Only NTU has it.
"2d13"  13 joints, x y in the image. Both datasets can give it:
            NTU   -> the 2D position of the Kinect joints in the colour video (stored in the .skeleton files)
            HMDB  -> a pose estimator (COCO keypoints) run on the video
        Both go through the SAME function `normalize_2d`, so the model sees the same kind of numbers for both.

The 13 shared joints (index: name):
     0 head     1 left shoulder    2 right shoulder    3 left elbow    4 right elbow    5 left wrist    6 right wrist
     7 left hip    8 right hip    9 left knee    10 right knee    11 left ankle    12 right ankle
"left" always means the person's own left side, in the Kinect skeleton and in the COCO keypoints.
Known small difference: joint 0 is the middle of the head in NTU and the nose in COCO.
"""
import numpy as np

JOINT_NAMES_13 = ["head", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
                  "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
HEAD, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 1, 2, 7, 8

# where each of the 13 joints is found in the other formats (positions counted from 0)
NTU25_TO_13 = [3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]       # NTU joints 4, 5, 9, 6, 10, 7, 11, 13, 17, 14, 18, 15, 19
COCO17_TO_13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]      # nose, then shoulders ... ankles (eyes and ears unused)

# bones as (joint, the joint it hangs on, towards the body centre), numbered from 1 like the NTU bones
BONES_25 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
BONES_13 = [(1, 2), (1, 3), (3, 2),                  # head - shoulders, shoulder - shoulder
            (4, 2), (6, 4), (5, 3), (7, 5),          # arms
            (8, 2), (9, 3), (9, 8),                  # hips - shoulders, hip - hip
            (10, 8), (12, 10), (11, 9), (13, 11)]    # legs

FORMATS = {
    "3d25": {"joints": 25, "channels": 3, "bones": BONES_25},
    "2d13": {"joints": 13, "channels": 2, "bones": BONES_13},
}


def normalize_2d(data):
    """Make 2D skeletons comparable between videos of any size and zoom.

    data: (T frames, 13 joints, 2 = x y in pixels, M persons). A joint that was not found is (0, 0).
    The origin moves to the middle of the main person's hips (average over the clip) and the unit of length becomes
    the main person's torso length (middle of shoulders to middle of hips, median over the clip).
    Joints that were not found stay exactly (0, 0). The second person keeps its position relative to the first.
    """
    data = data.astype(np.float32)
    found = np.abs(data).sum(axis=2) > 0                                          # (T, 13, M)
    torso_joints = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    torso_frames = found[:, torso_joints, 0].all(axis=1)                          # frames where the torso is complete
    if not torso_frames.any():
        raise ValueError("no frame shows both shoulders and both hips of the main person")
    main = data[torso_frames, :, :, 0]                                            # (frames, 13, 2)
    shoulders = (main[:, L_SHOULDER] + main[:, R_SHOULDER]) / 2
    hips = (main[:, L_HIP] + main[:, R_HIP]) / 2
    origin = hips.mean(axis=0)
    torso_length = np.median(np.linalg.norm(shoulders - hips, axis=1))
    if torso_length < 1e-3:
        raise ValueError("torso length is zero")
    normalized = (data - origin[None, None, :, None]) / torso_length
    return normalized * found[:, :, None, :]                                      # not-found joints back to (0, 0)