import numpy as np
import cv2

from sklearn.metrics import roc_curve, auc, precision_score, recall_score


def absolute_voxel_volume_difference(label_gt, label_pred):
    vox_volume_gt = label_gt.sum()
    vox_volume_pred = label_pred.sum()

    return np.absolute(vox_volume_gt - vox_volume_pred)


def roc_auc(label_gt, label_pred):
    y_true = np.array(label_gt).flatten()
    y_scores = np.array(label_pred).flatten()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc_score = auc(fpr, tpr)
    return roc_auc_score


def precision(label_gt, label_pred):
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()

    return precision_score(img_A, img_B)


def recall(label_gt, label_pred):
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()

    return recall_score(img_A, img_B)

def intersection_over_union(label_gt, label_pred):
    smooth = 1.0e-6
    iflat = np.array(label_pred).flatten()
    tflat = np.array(label_gt).flatten()
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum() - intersection
    return intersection / (union + smooth)


def dice_score(target, input):
    smooth = 1.0e-6
    iflat = np.array(input).flatten()
    tflat = np.array(target).flatten()
    intersection = (iflat * tflat).sum()

    return ((2. * intersection) /
            (iflat.sum() + tflat.sum() + smooth))


def distance_metric(seg_A, seg_B, dx, k):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """

    # Extract the label k from the segmentation maps to generate binary maps
    seg_A = (seg_A == k)
    seg_B = (seg_B == k)

    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd
