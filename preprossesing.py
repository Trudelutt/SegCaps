import os
import json
import csv
import pandas as pd
import SimpleITK as sitk
import numpy as np
#import matplotlib.pyplot as plt
#from batch_generator import generate_train_batches
from glob import glob
from os import mkdir
from os.path import join, basename
from tqdm import tqdm
from keras.utils import to_categorical
#from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import nibabel as nib
#import dicom2nifti
#from augmentation import dataaug



#Both LM and RCA
def make_both_label():
    path = glob("../st.Olav/*/*/*/")
    for i in xrange(len(path)):
        try:
            data_path = glob(path[i] + "*CCTA.nii.gz")[0]
            print(data_path)
            label_path = [glob(path[i] + "*LM.nii.gz")[0], glob(path[i] + "*RCA.nii.gz")[0]]
            sitk_label  = sitk.ReadImage(label_path[0], sitk.sitkFloat32)
            sitk_label  += sitk.ReadImage(label_path[1], sitk.sitkFloat32 )
            sitk.WriteImage(sitk_label, data_path.replace("CCTA", "both"))
        except:
            print("Could not make both file" + str(glob(path[i])))

def write_pridiction_to_file(label_array, prediction_array, tag, path="./predictions/prediction.nii.gz", label_path=None):
    meta_sitk = sitk.ReadImage(label_path)
    print(prediction_array.shape)
    sitk_image = sitk.GetImageFromArray(label_array)
    sitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(sitk_image, path.replace("nii", "gt.nii"))

    predsitk_image = sitk.GetImageFromArray(prediction_array)
    predsitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(predsitk_image, path)
    print("Writing prediction is done...")

def write_to_file(numpy_array, meta_path, path):
    print(path)
    meta_sitk = sitk.ReadImage(meta_path)
    sitk_image = sitk.GetImageFromArray(numpy_array[:meta_sitk.GetDepth()])
    sitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(sitk_image, path)

def write_all_labels(path):
    image = read_numpyarray_from_file(path+"LM.nii.gz")
    image += read_numpyarray_from_file(path+"Aorta.nii.gz")
    image += read_numpyarray_from_file(path+ "RCA.nii.gz")
    image[image == 2] = 1
    sitk_image = sitk.GetImageFromArray(image)
    sitk.WriteImage(sitk_image, "all_labels.nii.gz")


def preprosses_images(image, label_data):
    image -= np.min(image)
    image = image/ np.max(image)
    image -= np.mean(image)
    image = image / np.std(image)
    #image *= 255
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return image, label


def get_preprossed_numpy_arrays_from_file(image_path, label_path):
    sitk_image  = sitk.ReadImage(image_path)
    numpy_image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    sitk_label  = sitk.ReadImage(label_path,sitk.sitkUInt8 )
    numpy_label = sitk.GetArrayFromImage(sitk_label).astype(np.uint8)
    if '3Dircadb' in image_path:
        max_value = 700
        min_value = -300
        numpy_image[numpy_image> max_value] = max_value
        numpy_image[numpy_image < min_value] = min_value
    if not np.array_equal(np.unique(numpy_label), np.array([0,1])):
        #print("numpy is not binary mask")
        #numpy_label = numpy_label / np.max(numpy_label)
        #threshold = np.median(numpy_label)
        #print("UNique values")
        #print(np.unique(numpy_label))
        frangi_with_threshold = np.zeros(numpy_label.shape)
        frangi_with_threshold[np.where(numpy_label > 1)] = 1.0
        #print("it is suposed to be binary now")
        print(np.unique(frangi_with_threshold))
        #frangi_sitk = sitk.GetImageFromArray(frangi_with_threshold)
        #frangi_sitk.CopyInformation(sitk_image)
        #sitk.WriteImage(frangi_sitk, join('logs','mask_test', image_path.split("/")[-1]))
        return preprosses_images(numpy_image, frangi_with_threshold)
    return preprosses_images(numpy_image, numpy_label)


def remove_slices_with_just_background(image, label):
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    image_list = []
    label_list = []
    for i in xrange(image.shape[0]):
        if(1 in label[i]):
            if(i < first_non_backgroud_slice):
                first_non_backgroud_slice = i
            last_non_backgroud_slice = i
    if(first_non_backgroud_slice-2 < 0):
        resize_label =  label[first_non_backgroud_slice-1:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice-1:last_non_backgroud_slice+1]
    elif(first_non_backgroud_slice-1 < 0):
        resize_label =  label[first_non_backgroud_slice:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice:last_non_backgroud_slice+1]
    else:
        resize_label =  label[first_non_backgroud_slice-2:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice-2:last_non_backgroud_slice+1]


    return resize_image, resize_label

    #Channels must be an odd numbers
def add_neighbour_slides_training_data(image, label, stride=5, channels=5):
    padd = channels//2
    image_with_channels = np.zeros((image.shape[0], image.shape[1], image.shape[2], channels))
    zeros_image = np.zeros(image[0].shape)
    for i in range(image.shape[0]):
        if(i< padd * stride):
            count = padd * stride
            for channel in range(channels -(padd -i//stride)):
                #print(channels-channel-1, i + count)
                image_with_channels[i][...,channels - channel-1] = image[i+count]
                count -= stride

        elif i >= (image.shape[0]-(padd*stride)):
            count = - (padd * stride)
            for channel in range((image.shape[0]-i) +padd ):
                if (i + count) >= (image.shape[0]):
                    break
            #    print(channel, count+i)
                image_with_channels[i][...,channel] = image[i + count]
                count += stride
            #print("nFinished")
        else:
            count = - (padd * stride)
            for channel in range(channels):
                image_with_channels[i][...,channel] = image[i + count]
                count += stride
    return image_with_channels, label


def fetch_training_data_ca_files(data_root_dir,label="LM"):
    #path = glob("../st.Olav/*/*/*/")
    #path = glob("../../st.Olav/*/*/*/")
    if data_root_dir=="../st.Olav":
        data_root_dir += "/*/*/*/"
    path = glob(data_root_dir)
    training_data_files= list()
    for i in xrange(len(path)):
        try:
            data_path = glob(path[i] + "*CCTA.nii.gz")[0]
            label_path = glob(path[i] + "*" + label + ".nii.gz")[0]
        except IndexError:
            if label=="both" and i == 0:
                print("Makes both labels")
                make_both_label()
                label_path = glob(path[i] + "*" + label + ".nii.gz")[0]
            else:
                print("out of xrange for %s" %(path[i]))
        else:
            training_data_files.append(tuple([data_path, label_path]))
    return training_data_files


def fetch_training_data_portal_veins_files(data_root_dir,label):
    data_path = data_root_dir +"/*/"
    path = glob(data_path)
    print(path)
    training_data_files= list()
    for i in xrange(len(path)):
        data_path = glob(path[i]+ "PATIENT1.*.nii.gz")[0]
        label_path = data_path.replace("PATIENT", label )
        training_data_files.append(tuple([data_path, label_path]))
    return training_data_files


"""def convert_dicom_to_nify(label):
    for i in range(1,21):
        dicom2nifti.dicom_series_to_nifti("../3Dircadb1/3Dircadb1."+str(i)+"/LABELLED_DICOM", "../3Dircadb1/3Dircadb1."+str(i)+"/LABELLED1." + str(i) + ".nii.gz")
    for i in range(1,21):
        dicom2nifti.dicom_series_to_nifti("../3Dircadb1/3Dircadb1."+str(i)+"/PATIENT_DICOM", "../3Dircadb1/3Dircadb1."+str(i)+"/PATIENT1."+str(i) + ".nii.gz")
    for i in range(1,21):
        dicom2nifti.dicom_series_to_nifti("../3Dircadb1/3Dircadb1."+str(i)+"/MASKS_DICOM/"+label, "../3Dircadb1/3Dircadb1."+str(i)+"/"+ label+ "1." + str(i) +".nii.gz")"""



def get_train_and_label_numpy(number_of_slices, train_list, label_list, channels=5):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2], channels))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2], 1))
    index = 0
    for i in xrange(len(train_list)):
        with tqdm(total=train_list[i].shape[0], desc='Adds splice  from image ' + str(i+1) +"/" + str(len(train_list))) as t:
            for k in xrange(train_list[i].shape[0]):
                train_data[index] = train_list[i][k]
                label_data[index] = label_list[i][k]
                index += 1
                t.update()

    return train_data, label_data



#TODO make sure that index not out of bounds
def get_prediced_image_of_test_files(args,files, number, tag):
    element = files[number]
    print("Prediction on " + element[0])
    return get_slices(args,files[number:number+1], tag)



def get_train_data_slices(args, train_files, tag = "LM"):
    traindata = []
    labeldata = []
    count_slices = 0
    for element in train_files:
        print(element[0])
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label, stride= args.stride, channels=args.channels)
        resized_image, resized_label = remove_slices_with_just_background(i, l)

        count_slices += resized_image.shape[0]
        traindata.append(resized_image)
        labeldata.append(resized_label)
        #aug_img, mask = dataaug(resized_image, resized_label, intensityinterval= [0.8, 1.2], print_aug_images= True)
    train_data, label_data = get_train_and_label_numpy(count_slices, traindata, labeldata, channels=channels)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    #label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label_data


def get_slices(args, files, tag="LM"):
    input_data_list = []
    label_data_list = []
    count_slices = 0
    for element in files:
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        numpy_image = np.float32(numpy_image)
        numpy_image -= np.mean(numpy_image)
        numpy_image = numpy_image / np.std(numpy_image)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label,stride=args.stride, channels= args.channels)
        count_slices += i.shape[0]
        input_data_list.append(i)
        label_data_list.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, input_data_list, label_data_list)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    #label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label_data



def get_predict_patches(image_numpy, label_numpy):
    image_patch_list = []
    label_patch_list = []
    orginal_shape = image_numpy.shape
    padded_shape_z, padded_shape_y, padded_shape_x = orginal_shape
    if not orginal_shape[0] % 64 == 0:
        padded_shape_z = orginal_shape[0] + 64 - (orginal_shape[0] % 64)
    if not orginal_shape[1] % 64 == 0:
        padded_shape_y = orginal_shape[1] + 64 - (orginal_shape[1] % 64)
    if not orginal_shape[2] % 64 == 0:
        padded_shape_x = orginal_shape[2] + 64 - (orginal_shape[2] % 64)

    image_numpy_padded = np.zeros((padded_shape_z, padded_shape_y, padded_shape_x))
    image_numpy_padded[0:image_numpy.shape[0], 0:image_numpy.shape[1], 0:image_numpy.shape[2]] = image_numpy
    mask_padded = np.zeros((image_numpy_padded.shape[0], image_numpy_padded.shape[1], image_numpy_padded.shape[2], 1))
    mask_padded[0:image_numpy.shape[0], 0:image_numpy.shape[1], 0:image_numpy.shape[2]] = label_numpy
    for z in range(0, image_numpy_padded.shape[0],64):
        for y in range(0, image_numpy_padded.shape[1],64):
            for x in range(0,image_numpy_padded.shape[2],64):
                image_patch_list.append(image_numpy_padded[z:z+64, y:y+64, x:x+64])
                label_patch_list.append((mask_padded[z:z+64, y:y+64, x:x+64], (z, z+64, y, y+64, x, x+64)))
    image_patch = np.array(image_patch_list)
    new_shape = (image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], image_patch.shape[3], 1)
    return image_patch.reshape(new_shape), label_patch_list, mask_padded.shape


def get_patches(image_numpy, label_numpy, remove_only_background_patches=False):
    image_patch_list = []
    label_patch_list = []
    orginal_shape = image_numpy.shape
    padded_shape_z, padded_shape_y, padded_shape_x = orginal_shape
    if not orginal_shape[0] % 64 == 0:
        padded_shape_z = orginal_shape[0] + 64 - (orginal_shape[0] % 64)
    if not orginal_shape[1] % 64 == 0:
        padded_shape_y = orginal_shape[1] + 64 - (orginal_shape[1] % 64)
    if not orginal_shape[2] % 64 == 0:
        padded_shape_x = orginal_shape[2] + 64 - (orginal_shape[2] % 64)

    image_numpy_padded = np.zeros((padded_shape_z, padded_shape_y, padded_shape_x))
    image_numpy_padded[0:image_numpy.shape[0], 0:image_numpy.shape[1], 0:image_numpy.shape[2]] = image_numpy
    mask_padded = np.zeros((image_numpy_padded.shape[0], image_numpy_padded.shape[1], image_numpy_padded.shape[2], 1))
    mask_padded[0:image_numpy.shape[0], 0:image_numpy.shape[1], 0:image_numpy.shape[2]] = label_numpy
    for z in range(0, image_numpy_padded.shape[0],64):
        for y in range(0, image_numpy_padded.shape[1],64):
            for x in range(0,image_numpy_padded.shape[2],64):
                if remove_only_background_patches:
                    if np.array_equal(np.unique(mask_padded[z:z+64, y:y+64, x:x +64]), np.array([0])):
                        continue
                image_patch_list.append(image_numpy_padded[z:z+64, y:y+64, x:x+64])
                label_patch_list.append(mask_padded[z:z+64, y:y+64, x:x+64])
    return image_patch_list, label_patch_list, mask_padded.shape


def get_training_patches(train_files, label = "LM", remove_only_background_patches=False, return_shape=False):
    training_patches = []
    mask_patches = []
    count = 1
    with tqdm(total=len(train_files), desc='Adds patches  from image ' + str(count) +"/" + str(len(train_files))) as t:
        for element in train_files:
            numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
            img_patch, mask_patch, padded_shape  = get_patches(numpy_image, numpy_label, remove_only_background_patches)
            training_patches.extend(img_patch)
            mask_patches.extend(mask_patch)
            count += 1
            t.update()
    training_patch_numpy = np.array(training_patches)
    new_shape = (training_patch_numpy.shape[0],training_patch_numpy.shape[1],training_patch_numpy.shape[2],training_patch_numpy.shape[3], 1)
    new_shape_training_patch = training_patch_numpy.reshape(new_shape)
    if return_shape:
        return new_shape_training_patch, np.array(mask_patches).reshape(new_shape), padded_shape
    else:
        return new_shape_training_patch, np.array(mask_patches).reshape(new_shape)

def get_prediced_patches_of_test_file(test_files, i, label):
    element = test_files[i]
    print("Prediction on " + element[0])
    return get_training_patches(test_files[i:i+1], label, return_shape=True)

def from_patches_to_numpy(patches, shape):
    print(shape)
    reshape_patches = patches[...,0]
    print(reshape_patches.shape)
    image_numpy = np.zeros(shape[:-1])
    i = 0
    for z in range(0, shape[0],64):
        for y in range(0, shape[1],64):
            for x in range(0, shape[2],64):
                image_numpy[z:z+64, y:y+64, x:x+64] = reshape_patches[i]
                i += 1
                #print(z,y,x, i)
    if(i != patches.shape[0]):
        print("something is wrong with the patches to numpy converting")
        print(i, patches.shape[0])
    return image_numpy








if __name__ == "__main__":
    #convert_dicom_to_nify("portalvein")
    files = fetch_training_data_portal_veins_files("../3Dircadb1","portalvein")
    numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(files[0][0], files[0][1])
    print(numpy_image.shape, numpy_label.shape)
    #convert_dicom_to_nify(files[0])
    #create_split('../st.Olav', 'both')
    #get_data_files("../st.Olav", label="both")
    #train_files, val, test = get_train_val_test("both")
    #pred, lab = get_prediced_image_of_test_files(test, 0, "both")
    #img_slices, lab_slices = get_train_data_slices(train[:1])
    #print(len(fetch_training_data_ca_files("../st.Olav",label="both")))
    #get_training_patches(train_files[:2], label = "LM", remove_only_background_patches=False, return_shape=False)
    """print(test[0])
    x, y, orgshape = get_prediced_patches_of_test_file(test, 0, "both")
    label = from_patches_to_numpy(y, orgshape)
    img, org_label = get_preprossed_numpy_arrays_from_file(test[0][0], test[0][1])
    print(np.unique(np.equal(org_label, label[:org_label.shape[0]])))
    write_to_file(label, meta_path=test[0][0], path="./results/14.feb/" +str(basename(test[0][1])))"""
