import os
import cv2
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# def draw_point(img, landmarks):
#     img = np.array(img).copy()
#     h, w = img.shape[:2]
#     l = math.sqrt(h * h + w * w)
#     for idx, point in enumerate(landmarks):
#         point = np.around(point).astype(np.int)
#         pos = (point[0], point[1])
#         cv2.putText(img, str(idx), pos,
#                     fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#                     fontScale=l * 1e-3,
#                     color=(0, 0, 255),
#                     thickness=int(l / 500))
#         t_size = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, l * 1e-3 , int(l / 500))[0]
#         print(t_size)
#         cv2.circle(img, pos, max(2, int(l / 200)), color=(255, 0, 0), thickness=-1)
#     return Image.fromarray(img.astype(np.uint8))


def draw_point(img, landmarks, visible=None):
    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    if visible is None:
        visible = np.ones(landmarks.shape[:2]).astype(np.bool)

    w, h = img.size
    l = math.sqrt(h * h + w * w)

    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts', 'arial.ttf')
    font = ImageFont.truetype(font_path, int(l * 1e-3 * 25))
    
    for k, landmark in enumerate(landmarks):
        for idx, point in enumerate(landmark):
            if not visible[k, idx]:
                continue
            x, y = np.around(point).astype(np.int)
            draw = ImageDraw.Draw(img)
            r = max(2, int(l / 200))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
            draw.text((x, y), str(idx), fill=(0, 0, 255), font=font)

    if is_np:
        img = np.asarray(img)

    return img


def calc_pitch_yaw_roll(landmarks_2D, landmarks_3D, cam_w=256, cam_h=256):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    # assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    # landmarks_3D = np.float32([
    #     [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
    #     [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
    #     [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
    #     [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
    #     [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
    #     [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
    #     [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
    #     [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
    #     [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
    #     [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
    #     [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
    #     [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
    #     [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
    #     [0.000000, -7.415691, 4.070434],  # CHIN
    # ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix, camera_distortion)
    #Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;
    
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    # return map(lambda k: k[0], euler_angles) # euler_angles contain (pitch, yaw, roll)
    return euler_angles.T


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.view((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.view((batch_size, num_joints, 1))
    idx = idx.view((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).type(torch.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2).type(torch.float32)

    preds *= pred_mask
    return preds, maxvals


def heatmaps2points(batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)
    # print(coords[0])

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = torch.tensor([hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]]).to(batch_heatmaps.device).type(torch.float32)
                coords[n][p] += torch.sign(diff) * 0.25

    return coords, maxvals
