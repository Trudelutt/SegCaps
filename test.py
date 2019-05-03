'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about testing.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join, basename
from os import makedirs
import csv
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
from metrics import dc, jc, assd, precision, recall

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import print_summary

from load_3D_data import generate_test_batches
from preprossesing import get_training_patches, from_patches_to_numpy


def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    print('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0
    return raw_output

def remove_noise(raw_output):
    all_labels = measure.label(raw_output)
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        if props[0].area / props[1].area > 5:  # if the largest is way larger than the second largest
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            if props[1].area / props[2].area > 5:
                thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
                thresholded_mask[all_labels == props[1].label] = 1
            else:
                for i in range(len(props)):
                    #print(props[i].area)
                    if props[i].area > 500:
                        thresholded_mask[all_labels == props[i].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1

    #thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask

def create_and_write_viz_nii(name, meta_sitk, pred, gt):
    print("Write viz nii file...")
    predtion_copy = np.copy(pred)
    predtion_copy[predtion_copy > 0.] = 2.
    vis_image = gt + predtion_copy
    viz_sitk = sitk.GetImageFromArray(vis_image)
    viz_sitk.CopyInformation(meta_sitk)
    sitk.WriteImage(viz_sitk, name)
    print("Done write viz nii file...")
    vis_rgb = create_rgb_viz(vis_image)
    return vis_rgb

def create_rgb_viz(vis_image):
    vis_rgb = np.zeros((np.concatenate((vis_image.shape, (3,)))))
    vis_copy = np.copy(vis_image)
    vis_rgb[:,:,:,0][vis_copy == 1] = 1
    vis_rgb[:,:,:,1][vis_copy == 2] = 1
    vis_rgb[:,:,:,2][vis_copy == 3] = 1
    #print(np.unique(vis_rgb[:,:,:,2]))
    return vis_rgb



def creat_qual_figure(img_data, predtion, fig_out_dir, img_path):
        # Plot Qual Figure
        print('Creating Qualitative Figure for Quick Reference')
        f, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(img_data[img_data.shape[0] // 3, :, :], alpha=1, cmap='gray')
        ax[0].imshow(predtion[img_data.shape[0] // 3, :, :], alpha=0.2, cmap='gray')
        #ax[0].imshow(gt_data[img_data.shape[0] // 3, :, :], alpha=0.2, cmap='Reds')
        ax[0].set_title('Slice {}/{}'.format(img_data.shape[0] // 3, img_data.shape[0]))
        ax[0].axis('off')

        ax[1].imshow(img_data[img_data.shape[0] // 2, :, :], alpha=1, cmap='gray')
        ax[1].imshow(predtion[img_data.shape[0] // 2, :, :], alpha=0.2, cmap='gray')
        #ax[1].imshow(gt_data[img_data.shape[0] // 2, :, :], alpha=0.2, cmap='Reds')
        ax[1].set_title('Slice {}/{}'.format(img_data.shape[0] // 2, img_data.shape[0]))
        ax[1].axis('off')

        ax[2].imshow(img_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=1, cmap='gray')
        ax[2].imshow(predtion[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.2, cmap='gray')
        ax[2].set_title(
            'Slice {}/{}'.format(img_data.shape[0] // 2 + img_data.shape[0] // 4, img_data.shape[0]))
        ax[2].axis('off')

        fig = plt.gcf()
        title = basename(img_path[1][:-7]).split("_Seg")[0]
        if "Frangi" in img_path[1]:
            title += "_Frangi"
        if "post" in img_path[1]:
            title += "_post"
        fig.suptitle(title)

        plt.savefig(join(fig_out_dir, basename(img_path[0][:-7]) + '_qual_fig' + '.png'),
                    format='png', bbox_inches='tight')
        plt.close('all')

def test(args, test_list, model_list, net_input_shape):
    if args.weights_path == '':
        weights_path = join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5')
        sub_res_weights_path = weights_path
    else:
        weights_path = args.weights_path
        sub_res_weights_path = join(args.check_dir, basename(weights_path.replace("saved_models/", "")))

    output_dir = join( 'results','split'+str(args.split_nr), sub_res_weights_path[:-5])
    raw_out_dir = join(output_dir, 'raw_output')
    fin_out_dir = join(output_dir, 'final_output')
    fig_out_dir = join(output_dir, 'qual_figs')
    try:
        makedirs(raw_out_dir)
    except:
        pass
    try:
        makedirs(fin_out_dir)
    except:
        pass
    try:
        makedirs(fig_out_dir)
    except:
        pass

    if len(model_list) > 1:
        eval_model = model_list[1]
    else:
        eval_model = model_list[0]
    """try:
        eval_model.load_weights(weights_path)
    except:
        print('Unable to find weights path. Testing with random weights.')"""
    eval_model.load_weights(weights_path)
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])

    # Set up placeholders
    outfile = ''
    if args.compute_dice:
        dice_arr = np.zeros((len(test_list)))
        dice_arr_post = np.zeros((len(test_list)))
        outfile += 'dice_'
    if args.compute_jaccard:
        jacc_arr = np.zeros((len(test_list)))
        outfile += 'jacc_'
    if args.compute_assd:
        assd_arr = np.zeros((len(test_list)))
        outfile += 'assd_'
    if args.compute_recall:
        recall_arr = np.zeros((len(test_list)))
        recall_arr_post = np.zeros((len(test_list)))
    if args.compute_precision:
        precision_arr = np.zeros((len(test_list)))
        precision_arr_post = np.zeros((len(test_list)))

    # Testing the network
    print('Testing... This will take some time...')

    with open(join(output_dir, args.save_prefix + outfile + 'scores.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ['Scan Name']
        if args.compute_dice:
            row.append('Dice Coefficient')
            row.append('Dice Coefficient Post')
        if args.compute_jaccard:
            row.append('Jaccard Index')
        if args.compute_assd:
            row.append('Average Symmetric Surface Distance')
        if args.compute_recall:
            row.append('Recall')
            row.append('Recall Post')
        if args.compute_precision:
            row.append('Precision')
            row.append('Precision Post')

        writer.writerow(row)
        for i, img in enumerate(tqdm(test_list)):
            #TODO this must change
            print(img[0])
            sitk_img = sitk.ReadImage(img[0])
            img_data = sitk.GetArrayFromImage(sitk_img)

            if args.net == 'bvnet3d':
                num_slices_before_padding = img_data.shape[0]
                pred_img, pred_mask, orgshape = get_training_patches([[img[0], img[1]]], args.label, remove_only_background_patches=False, return_shape=True)
                num_slices = pred_mask.shape[0]
            else:
                num_slices = img_data.shape[0]

            output_array = eval_model.predict_generator(generate_test_batches(args,args.label, args.data_root_dir, [img],
                                                                              net_input_shape,
                                                                              batchSize=1,
                                                                              numSlices=args.slices,
                                                                              subSampAmt=0,
                                                                              stride=1),
                                                        steps=num_slices, max_queue_size=1, workers=1,
                                                        use_multiprocessing=False, verbose=1)

            if args.net.find('caps') != -1:
                output = output_array[0][:,:,:,0]
                #recon = output_array[1][:,:,:,0]
            elif args.net == 'bvnet3d':
                output = from_patches_to_numpy(output_array, orgshape)
                output= output[:num_slices_before_padding]
                output_array = output.reshape(img_data.shape)

            else:
                output = output_array[:,:,:,0]

            output_img = sitk.GetImageFromArray(output)
            print('Segmenting Output')
            threshold_output = threshold_mask(output, args.thresh_level)
            output_raw_mask = sitk.GetImageFromArray(threshold_output)
            output_bin = remove_noise(threshold_output)
            output_mask = sitk.GetImageFromArray(output_bin)

            output_img.CopyInformation(sitk_img)
            output_raw_mask.CopyInformation(sitk_img)
            output_mask.CopyInformation(sitk_img)


            print('Saving Output')
            sitk.WriteImage(output_raw_mask, join(raw_out_dir, basename(img[1][:-7]) + '_raw_output' + img[1][-7:]))
            sitk.WriteImage(output_mask, join(fin_out_dir, basename(img[1][:-7]) + '_final_output' + img[1][-7:]))

            # Load gt mask
            #TODO change to get correcr mask name
            sitk_mask = sitk.ReadImage(img[1])
            gt_data = sitk.GetArrayFromImage(sitk_mask)
            gt_data[gt_data == 255] = 1
            post_prediction = create_and_write_viz_nii(join(raw_out_dir, basename(img[1][:-7]) + '_final_output_viz' + img[1][-7:]), sitk_img, output_bin, gt_data)
            raw_prediction = create_and_write_viz_nii(join(raw_out_dir, basename(img[1][:-7]) + '_raw_output_viz' + img[1][-7:]), sitk_img,threshold_output , gt_data)


            creat_qual_figure(img_data, raw_prediction, fig_out_dir , img)
            creat_qual_figure(img_data, post_prediction, fig_out_dir ,  (img[0].replace('CCTA', 'CCTA_post'), img[1].replace(args.label, args.label + '_post')))
            if args.frangi_mode == "frangi_comb":
                sitk_frangi = sitk.ReadImage(img[0].replace('CCTA', 'CCTA_Frangi'))
                frangi_data = sitk.GetArrayFromImage(sitk_frangi)
                creat_qual_figure(frangi_data, raw_prediction, fig_out_dir , (img[0].replace('CCTA', 'CCTA_Frangi'), img[1].replace(args.label, args.label + '_Frangi')))
                creat_qual_figure(frangi_data, post_prediction, fig_out_dir , (img[0].replace('CCTA', 'CCTA_Frangi_post'), img[1].replace(args.label, args.label + '_Frangi_post')))



            row = [img[0][:-7]]
            if args.compute_dice:
                print('Computing Dice')
                dice_arr[i] = dc(threshold_output, gt_data)
                print('\tDice: {}'.format(dice_arr[i]))
                row.append(dice_arr[i])
                print('Computing Dice Post')
                dice_arr_post[i] = dc(output_bin, gt_data)
                print('\tDice post: {}'.format(dice_arr_post[i]))
                row.append(dice_arr_post[i])
            if args.compute_jaccard:
                print('Computing Jaccard')
                jacc_arr[i] = jc(output_bin, gt_data)
                print('\tJaccard: {}'.format(jacc_arr[i]))
                row.append(jacc_arr[i])
            if args.compute_assd:
                print('Computing ASSD')
                assd_arr[i] = assd(output_bin, gt_data, voxelspacing=sitk_img.GetSpacing(), connectivity=1)
                print('\tASSD: {}'.format(assd_arr[i]))
                row.append(assd_arr[i])
            if args.compute_recall:
                print('Recall')
                recall_arr[i] = recall(threshold_output, gt_data)
                print('\tRecall: {}'.format(recall_arr[i]))
                row.append(recall_arr[i])
                print('Recall Post')
                recall_arr_post[i] = recall(output_bin, gt_data)
                print('\tRecall post: {}'.format(recall_arr_post[i]))
                row.append(recall_arr_post[i])
            if args.compute_precision:
                print('Precision')
                precision_arr[i] = precision(threshold_output, gt_data)
                print('\tPrecision: {}'.format(precision_arr[i]))
                row.append(precision_arr[i])
                print('Precision Post')
                precision_arr_post[i] = precision(output_bin, gt_data)
                print('\tPrecision post: {}'.format(precision_arr_post[i]))
                row.append(precision_arr_post[i])

            writer.writerow(row)

        row = ['Average Scores']
        if args.compute_dice:
            row.append(np.mean(dice_arr))
            row.append(np.mean(dice_arr_post))
        if args.compute_jaccard:
            row.append(np.mean(jacc_arr))
        if args.compute_assd:
            row.append(np.mean(assd_arr))
        if args.compute_recall:
            row.append(np.mean(recall_arr))
            row.append(np.mean(recall_arr_post))
        if args.compute_precision:
            row.append(np.mean(precision_arr))
            row.append(np.mean(precision_arr_post))
        writer.writerow(row)

    print('Done.')
