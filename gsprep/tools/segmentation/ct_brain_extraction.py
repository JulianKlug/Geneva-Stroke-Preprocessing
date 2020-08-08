import os
import argparse
import numpy as np
import nibabel as nib
import nipype.interfaces.fsl as fsl
import tempfile
import shutil

def ct_brain_extraction(data, working_directory=None, fractional_intensity_threshold = 0.01, save_output = False):
    """
    Automatic brain extraction of non contrast head CT images using bet2 by fsl.
    Ref.: Muschelli J, Ullman NL, Mould WA, Vespa P, Hanley DF, Crainiceanu CM. Validated automatic brain extraction of head CT images. NeuroImage. 2015 Jul 1;114:379–85.

    :param data: [str; np.ndarray] path to input data or input data in form of np.ndarray (x, y, z)
    :param working_directory: [str] path to directory to use to save temporary files and final output files
    :param fractional_intensity_threshold: fractional intensity threshold (0->1); default=0.01; smaller values give larger brain outline estimates
    :param save_output: [boolean] save or discard output
    :return: brain_mask, masked_image: np.ndarray
    """

    temp_files = []
    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    if isinstance(data, np.ndarray):
        data_path = os.path.join(working_directory, 'temp_bet_input.nii.gz')
        data_img = nib.Nifti1Image(data.astype('float64'))
        nib.save(data_img, data_path)
        temp_files.append(data_path)
    elif os.path.exists(data):
        data_path = data
    else:
        raise NotImplementedError('Data has to be a path or an np.ndarray')

    output_file = os.path.join(working_directory, 'bet_output.nii.gz')
    output_mask_file = os.path.join(working_directory, 'bet_output_mask.nii.gz')
    if not save_output:
        temp_files.append(output_file)
        temp_files.append(output_mask_file)
    temp_intermediate_file = os.path.join(working_directory, 'temp_intermediate_file.nii.gz')
    temp_files.append(temp_intermediate_file)


    # Thresholding Image to 0-100
    # cli: fslmaths "${img}" -thr 0.000000 -uthr 100.000000  "${outfile}"
    thresholder1 = fsl.Threshold()
    thresholder1.inputs.in_file = data_path
    thresholder1.inputs.out_file = output_file
    thresholder1.inputs.thresh = 0
    thresholder1.inputs.direction = 'below'
    thresholder1.run()

    thresholder2 = fsl.Threshold()
    thresholder2.inputs.in_file = output_file
    thresholder2.inputs.out_file = output_file
    thresholder2.inputs.thresh = 100
    thresholder2.inputs.direction = 'above'
    thresholder2.run()

    # Creating 0 - 100 mask to remask after filling
    # cli: fslmaths "${outfile}"  -bin   "${tmpfile}";
    # cli: fslmaths "${tmpfile}.nii.gz" -bin -fillh "${tmpfile}"
    binarizer1 = fsl.UnaryMaths()
    binarizer1.inputs.in_file = output_file
    binarizer1.inputs.out_file = temp_intermediate_file
    binarizer1.inputs.operation = 'bin'
    binarizer1.run()

    binarizer2 = fsl.UnaryMaths()
    binarizer2.inputs.in_file = temp_intermediate_file
    binarizer2.inputs.out_file = temp_intermediate_file
    binarizer2.inputs.operation = 'bin'
    binarizer2.run()

    fill_holes1 = fsl.UnaryMaths()
    fill_holes1.inputs.in_file = temp_intermediate_file
    fill_holes1.inputs.out_file = temp_intermediate_file
    fill_holes1.inputs.operation = 'fillh'
    fill_holes1.run()

    # Presmoothing image
    # cli: fslmaths "${outfile}" - s 1 "${outfile}"
    smoothing = fsl.IsotropicSmooth()
    smoothing.inputs.in_file = output_file
    smoothing.inputs.out_file = output_file
    smoothing.inputs.sigma = 1
    smoothing.run()

    # Remasking Smoothed Image
    # cli: fslmaths "${outfile}" - mas "${tmpfile}"  "${outfile}"
    masking1 = fsl.ApplyMask()
    masking1.inputs.in_file = output_file
    masking1.inputs.out_file = output_file
    masking1.inputs.mask_file = temp_intermediate_file
    masking1.run()

    # Running bet2
    # cli: bet2 "${outfile}" "${outfile}" - f ${intensity} - v
    btr = fsl.BET()
    btr.inputs.in_file = output_file
    btr.inputs.out_file = output_file
    btr.inputs.frac = fractional_intensity_threshold
    btr.run()

    # Using fslfill to fill in any holes in mask
    # cli: fslmaths "${outfile}" - bin - fillh "${outfile}_Mask"
    binarizer3 = fsl.UnaryMaths()
    binarizer3.inputs.in_file = output_file
    binarizer3.inputs.out_file = output_mask_file
    binarizer3.inputs.operation = 'bin'
    binarizer3.run()

    fill_holes2 = fsl.UnaryMaths()
    fill_holes2.inputs.in_file = output_mask_file
    fill_holes2.inputs.out_file = output_mask_file
    fill_holes2.inputs.operation = 'fillh'
    fill_holes2.run()

    # Using the filled mask to mask original image
    # cli: fslmaths "${img}" -mas "${outfile}_Mask"  "${outfile}"
    masking2 = fsl.ApplyMask()
    masking2.inputs.in_file = data_path
    masking2.inputs.out_file = output_file
    masking2.inputs.mask_file = output_mask_file
    masking2.run()

    brain_mask = nib.load(output_mask_file).get_fdata()
    masked_image = nib.load(output_file).get_fdata()

    # delete temporary files
    for file in temp_files:
        os.remove(file)

    if not save_output:
        shutil.rmtree(working_directory)

    return brain_mask, masked_image



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic brain extraction of non contrast head CT images using bet2 by fsl.')
    parser.add_argument('data_path')
    parser.add_argument('-w', '--working_directory',  help='Working directory', required=True, default=None)
    parser.add_argument('-f', '--fractional_intensity_threshold',  help='Smaller values give larger brain outline estimates [0,1]', required=False, default=0.01)
    parser.add_argument('-s', '--save_output', nargs='?', const=True, default=False, required=False)

    args = parser.parse_args()
    ct_brain_extraction(args.data_path, args.working_directory, args.fractional_intensity_threshold, args.save_output)


