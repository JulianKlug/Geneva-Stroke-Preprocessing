import os
import nibabel as nib
import pandas as pd
from gsprep.evaluation.metrics import dice_score, absolute_voxel_volume_difference, roc_auc, precision, recall, \
    intersection_over_union, distance_metric
from tqdm import tqdm


def grader(label_dir: str, pred_dir: str, output_dir: str = None):
    subjects = [o for o in os.listdir(label_dir)
                if o.endswith('.nii') or o.endswith('.nii.gz')]

    results_columns = ['subject', 'dice', 'absolute_voxel_volume_difference', 'roc_auc', 'precision', 'recall',
                       'intersection_over_union', 'mean_distance', 'Hausdorff_distance']

    results_df = pd.DataFrame(columns=results_columns)

    for subject in tqdm(subjects):
        gt = nib.load(os.path.join(label_dir, subject)).get_fdata()
        pred = nib.load(os.path.join(pred_dir, subject)).get_fdata()

        mean_distance, hausdorff_distance = distance_metric(gt, pred, dx=2.00, k=1)

        results_df = results_df.append(pd.DataFrame([[
            subject,
            dice_score(gt, pred),
            absolute_voxel_volume_difference(gt, pred),
            roc_auc(gt, pred),
            precision(gt, pred),
            recall(gt, pred),
            intersection_over_union(gt, pred),
            mean_distance,
            hausdorff_distance
        ]], columns=results_columns
        ))

    if output_dir is None:
        output_dir = pred_dir

    summary_df = results_df.describe()

    writer = pd.ExcelWriter(os.path.join(output_dir, 'grader_results.xlsx'))
    results_df.to_excel(writer, 'individual')
    summary_df.to_excel(writer, 'summary')
    writer.save()

grader('/Users/jk1/temp/nnunet_3d_fullres_eval/labelsTs', '/Users/jk1/temp/nnunet_3d_fullres_eval/3d_fullres_nnunet_pred')


