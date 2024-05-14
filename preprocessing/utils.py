# -----------------------------------------------------------------------------
# This file is created as part of the multi-planar prostate segmentation project
#
#  file:           utils.py
#  author:         Gino Gulamhussene, Anneke Meyer, Otto-von-Guericke University Magdeburg
#  year:           2017
#
# -----------------------------------------------------------------------------


import os
import numpy as np
import math
import sys
import SimpleITK as sitk
import time   
import glob
import csv
import json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard, TerminateOnNaN
from keras.callbacks import CSVLogger
import logging
from tqdm import tqdm
import preprocessing.preprocess as prep
import SimpleITK as sitk
import matplotlib 
import matplotlib.pyplot as plt
import time
from deeds import registration
sys.path.append('/home/gino/projects/4D_MRI_CNN')
import preprocessing.img_processing as img_p

############################ utils functions ##############################

def make_hp_iterations_for_parameter(hp_base, parameter_name, values):
    result = []
    for value in values:
        hp_temp = hp_base.copy()
        hp_temp[parameter_name] = value
        result.append(hp_temp)
    return result

class EarlyStopping_Thrsh(Callback):
    def __init__(self, monitor='val_loss', max_val=100.0):
        self.monitor = monitor
        self.max_val = max_val

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current > self.max_val:
            self.model.stop_training = True 


            
def make_callbacks(hp, csv_filename='', model_filename=''):
    result = []

    ### csv logger callback
    if csv_filename != '':
        csv_logger = CSVLogger(csv_filename, append=True, separator=';')
        result.append(csv_logger)

    ### model checkpoint callback
    if model_filename != '':
        model_checkpoint = ModelCheckpoint(model_filename, monitor='val_mse', save_best_only=True, verbose=1)
        result.append(model_checkpoint)

    ### earlyStopImprovement callback
    earlyStopImprovement = EarlyStopping(monitor='val_mse', min_delta = hp['ERLSTOP_MIN_DELTA'], patience = hp['ERLSTOP_PATIENCE'], verbose=1, mode='min')
    result.append(earlyStopImprovement)
    earlyStop_Thrsh = EarlyStopping_Thrsh(monitor='val_mse', max_val=100.0)
    result.append(earlyStop_Thrsh)
    
    ### terminate on nan
    result.append(TerminateOnNaN())
    
    ### ReduceLROnPlateau callback
    LRDecay = ReduceLROnPlateau(monitor='val_mse', factor = hp['LR_DECAY_FACTOR'], patience = hp['LR_DECAY_PATIENCE'], verbose=1, mode='min', min_lr = hp['LR_DECAY_MIN_LR'], min_delta = hp['LR_DECAY_MIN_DELTA'])
    result.append(LRDecay)
    
    return result

# ===================================================
# create a function level logger
# ===================================================
def function_logger(name, file_level = logging.DEBUG, console_level = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) #By default, logs all messages

    if console_level != None:
        ch = logging.StreamHandler() #StreamHandler logs to console
        ch.setLevel(console_level)
        ch_format = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(ch_format)
        logger.addHandler(ch)

    fh = logging.FileHandler("{0}.log".format(name))
    fh.setLevel(file_level)
    fh_format = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)-8s - %(message)s')
    fh.setFormatter(fh_format)
    logger.addHandler(fh)

    return logger

def log_time(start, name, d, n):
    end = time.time()
    #print(name, 'took ', str(end - start), 's.')
    d.append(end - start)
    n.append(name)
    return end

def save_experiment_json(hyper_parameters, save_path, save_name=''):
    if save_name == '':
        save_name = 'doc.json'
    with open(os.path.join(save_path, save_name), 'w') as fp:
        json.dump(hyper_parameters, fp, indent=4)

def load_experiment_json(path, name=''):
    if name != '':
        path = os.path.join(path, name)
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data
    
def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getMeanAndStd(inputDir):

    patients = os.listdir(inputDir)
    list = []
    for patient in patients:
        data = os.listdir(inputDir + '/' + patient)
        for imgName in data:
            if 'tra' in imgName or 'cor' in imgName or 'sag' in imgName:
                img = sitk.ReadImage(inputDir + '/' + patient + '/' + imgName)
                arr = sitk.GetArrayFromImage(img)
                arr = np.ndarray.flatten(arr)

                list.append(np.ndarray.tolist(arr))


    array = np.concatenate(list).ravel()
    mean = np.mean(array)
    std = np.std(array)
    print(mean, std)
    return mean, std


def normalizeByMeanAndStd(img, mean, std):

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    img = castImageFilter.Execute(img)
    subFilter = sitk.SubtractImageFilter()
    image = subFilter.Execute(img, mean)

    divFilter = sitk.DivideImageFilter()
    image = divFilter.Execute(image, std)

    return image
	
	
def changeSizeWithPadding(inputImage, newSize):

    inSize = inputImage.GetSize()
    array = sitk.GetArrayFromImage(inputImage)
    minValue = array.min()

    # print("newSize", newSize[0], newSize[1], newSize[2])

    nrVoxelsX = max(0,int((newSize[0] - inSize[0]) / 2))
    nrVoxelsY = max(0,int((newSize[1] - inSize[1]) / 2))
    nrVoxelsZ = max(0,int((newSize[2] - inSize[2]) / 2))


    if nrVoxelsX == 0 and nrVoxelsY == 0 and nrVoxelsZ == 0:
        print("sameSize")
        return inputImage

    print("Padding Constant Value", minValue)
    upperBound = [nrVoxelsX,nrVoxelsY,nrVoxelsZ]
    filter = sitk.ConstantPadImageFilter()
    filter.SetPadLowerBound([nrVoxelsX, nrVoxelsY, nrVoxelsZ])
    if inSize[0] % 2==1:
        upperBound[0] = nrVoxelsX+1
    if inSize[1] % 2 == 1:
        upperBound[1] = nrVoxelsY + 1
    if inSize[2] % 2 == 1 and nrVoxelsZ!=0:
        upperBound[2] = nrVoxelsZ + 1

    filter.SetPadUpperBound(upperBound)
    filter.SetConstant(int(minValue))
    outPadding = filter.Execute(inputImage)
    print("outSize after Padding", outPadding.GetSize())

    return outPadding


# normlaize intensities according to the 99th and 1st percentile of the input image intensities
def normalizeIntensitiesPercentile(*imgs):

    out = []

    for img in imgs:
        array = np.ndarray.flatten(sitk.GetArrayFromImage(img))

        upperPerc = np.percentile(array, 99) #98
        lowerPerc = np.percentile(array, 1) #2
        print('percentiles', upperPerc, lowerPerc)

        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        normalizationFilter = sitk.IntensityWindowingImageFilter()
        normalizationFilter.SetOutputMaximum(1.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        floatImg = castImageFilter.Execute(img)
        outNormalization = normalizationFilter.Execute(floatImg)
        out.append(outNormalization)

    return out


def getMaximumValue(img):

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    return maxValue

def thresholdImage(img, lowerValue, upperValue, outsideValue):

    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetUpper(upperValue)
    thresholdFilter.SetLower(lowerValue)
    thresholdFilter.SetOutsideValue(outsideValue)

    out = thresholdFilter.Execute(img)
    return out



def binaryThresholdImage(img, lowerThreshold):

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)

    return thresholded



def resampleImage(inputImage, newSpacing, interpolator, defaultValue):

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing= inputImage.GetSpacing()
    newWidth = oldSpacing[0]/newSpacing[0]* oldSize[0]
    newHeight = oldSpacing[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    minFilter = sitk.StatisticsImageFilter()
    minFilter.Execute(inputImage)
    minValue = minFilter.GetMinimum()

    filter = sitk.ResampleImageFilter()
    inputImage.GetSpacing()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(inputImage.GetOrigin())
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(newSize)
    filter.SetDefaultPixelValue(defaultValue)
    outImage = filter.Execute(inputImage)

    return outImage


def resampleToReference(inputImg, referenceImg, interpolator, defaultValue):

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImg = castImageFilter.Execute(inputImg)

    #sitk.WriteImage(inputImg,'input.nrrd')

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImg)
    filter.SetDefaultPixelValue(float(defaultValue)) ## -1
    # float('nan')
    filter.SetInterpolator(interpolator)

    outImage = filter.Execute(inputImg)


    return outImage


def castImage(img, type):

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(type) #sitk.sitkUInt8
    out = castFilter.Execute(img)

    return out

# corrects the size of an image to a multiple of the factor
def sizeCorrectionImage(img, factor, imgSize):
# assumes that input image size is larger than minImgSize, except for z-dimension
# factor is important in order to resample image by 1/factor (e.g. due to slice thickness) without any errors
    size = img.GetSize()
    correction = False
    # check if bounding box size is multiple of 'factor' and correct if necessary
    # x-direction
    if (size[0])%factor != 0:
        cX = factor-(size[0]%factor)
        correction = True
    else:
        cX = 0
    # y-direction
    if (size[1])%factor != 0:
        cY = factor-((size[1])%factor)
        correction = True
    else:
        cY  = 0

    if (size[2]) !=imgSize:
        cZ = (imgSize-size[2])
        # if z image size is larger than maxImgsSize, crop it (customized to the data at hand. Better if ROI extraction crops image)
        if cZ <0:
            print('image gets filtered')
            cropFilter = sitk.CropImageFilter()
            cropFilter.SetUpperBoundaryCropSize([0,0,int(math.floor(-cZ/2))])
            cropFilter.SetLowerBoundaryCropSize([0,0,int(math.ceil(-cZ/2))])
            img = cropFilter.Execute(img)
            cZ=0
        else:
            correction = True
    else:
        cZ = 0

    # if correction is necessary, increase size of image with padding
    if correction:
        filter = sitk.ConstantPadImageFilter()
        filter.SetPadLowerBound([int(math.floor(cX/2)), int(math.floor(cY/2)), int(math.floor(cZ/2))])
        filter.SetPadUpperBound([math.ceil(cX/2), math.ceil(cY), math.ceil(cZ/2)])
        filter.SetConstant(0)
        outPadding = filter.Execute(img)
        return outPadding

    else:
        return img


def resampleToReference_shapeBasedInterpolation(inputGT, referenceImage, backgroundValue =0):

    filter =sitk.SignedMaurerDistanceMapImageFilter()
    filter.SetUseImageSpacing(True)
    filter.UseImageSpacingOn()
    filter.SetInsideIsPositive(True)
    dist = filter.Execute(castImage(inputGT, sitk.sitkUInt8))
    dist = resampleToReference(dist, referenceImage, sitk.sitkLinear, -100)
    GT = binaryThresholdImage(dist, 0)

    return GT


def getBoundingBox(img):

   # masked = binaryThresholdImage(img, 0.1)
   ### ToDo: masked is needed for getting bounding box from intrsecting tra, sag, cor volumes. Is it disturing for other cases, e.g. evaluation?
    masked = binaryThresholdImage(img, 0.001)

    img = castImage(masked, sitk.sitkUInt8)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(img)

    bb = statistics.GetBoundingBox(1)

    return bb

def getLargestConnectedComponents(img):

    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(img)

    labelStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelStatistics.Execute(connectedComponents)
    nrLabels = labelStatistics.GetNumberOfLabels()
    print('Nr Of Labels:', nrLabels)

    if nrLabels ==1:
        return img

    biggestLabelSize = 0
    biggestLabelIndex = 1
    for i in range(1, nrLabels+1):
        curr_size = labelStatistics.GetNumberOfPixels(i)
        if curr_size > biggestLabelSize:
            biggestLabelSize = curr_size
            biggestLabelIndex = i

    largestComponent = sitk.BinaryThreshold(connectedComponents, biggestLabelIndex, biggestLabelIndex)

    return largestComponent

def cropImage(img, lowerBound, upperBound):

    cropFilter = sitk.CropImageFilter()
    cropFilter.SetUpperBoundaryCropSize(upperBound)
    cropFilter.SetLowerBoundaryCropSize(lowerBound)
    img = cropFilter.Execute(img)

    return img

def makedir(dirpath):
  try:
    os.makedirs(dirpath)
  except OSError:
    # [Errno 17] File exists
    pass

def calculate_sample_weights(train_files_dir, case, train_id_list):

    if os.path.isfile(os.path.join(train_files_dir, case, 'sample_weights_.npy')):
        dist_arr = np.load(os.path.join(train_files_dir, case, 'sample_weights_.npy'))
    else:

        dist_list = []
        # for element in train_id_list:
        print('Calculating sample weights for: ', case)
        for id in tqdm(train_id_list):
            d = sitk.ReadImage(id)
            n = sitk.ReadImage(id.replace('_D_', '_N_'))
            x1 = d.GetOrigin()[0]  # get origin (x) of element
            x2 = n.GetOrigin()[0]  # get origin (x) of corresp. navigation slice
            dist = (abs(x1-x2))  #compute distance in between
            dist_list.append(dist)
        dist_arr = np.array(dist_list)
        # normalize array
        max = np.amax(dist_arr)
        dist_arr = dist_arr / max
        dist_arr = dist_arr + 0.1

        np.save(os.path.join(train_files_dir, case, 'sample_weights_.npy'), dist_arr)

    weight_list = dist_arr.tolist()
    print(len(weight_list))

    return weight_list

def auto_predict_3D(model, generator, batch_num=1, max_vols=10, max_slices=30, out_path='', save_name_pref=''): 
    for i in range(batch_num):
        refVols = []
        tic = time.perf_counter()
        (X_val, Y_val, Weights) = generator.__getitem__(index=i, return_RefVol=True, ref_vols=refVols)
        prediction = model.predict(X_val)
        tic = time.perf_counter()
        print(f"volume reconstruction took {toc - tic:0.4f} seconds")

        bs = prediction.shape[0]
        for j in range(bs):
            if max_vols <= 0:
                break
            res_img_arr = prediction[j,:,:,:,0]
            res_img = sitk.GetImageFromArray(res_img_arr)
            res_img.CopyInformation(refVols[0])

            result_name = save_name_pref + '_pred_3D_{}.nrrd'.format(j + i*bs)
            sitk.WriteImage(res_img, os.path.join(out_path, result_name))
            max_vols -= 1
        
        for j in range(bs):
            if max_slices <= 0:
                break
            res_img_arr = prediction[j,:,:,:,0]
            result_crosssection = np.fliplr(np.flipud(res_img_arr[:,31,:]))         
            file_name = save_name_pref + '_crossSection_{}.png'.format(j + i*bs)
            matplotlib.image.imsave(os.path.join(out_path, file_name), result_crosssection, cmap="gray")
            max_slices -= 1
    
def visualize_sample(X, Y, P, skip=0, out_path='', save_name_pref=''):  
  s = X.shape[0]
  x = X.shape[1]
  y = X.shape[2]
  for i in range(0,s,skip + 1):
    result = np.zeros((x,y*6), dtype=np.float32)
    result[:,:y]      = np.fliplr(np.flipud(     X[i,:,:,0]))
    result[:,y:y*2]   = np.fliplr(np.flipud(     X[i,:,:,1]))
    result[:,y*2:y*3] = np.fliplr(np.flipud(     X[i,:,:,2]))
    # result[:,y*3:y*4] = np.fliplr(np.flipud(     X[i,:,:,3]))
    result[:,y*4:y*5] = np.fliplr(np.flipud(     P[i,:,:,0]))
    result[:,y*5:y*6] = np.fliplr(np.flipud(     Y[i,:,:,0]))
    
    name = save_name_pref + str(i) + '.png'
    matplotlib.image.imsave( os.path.join(out_path, name), result, cmap="gray")
        
def save_groundTruth_prediction_pairs(Y, P, skip=0, out_path='', save_name_pref=''):  
  s = Y.shape[0]
  x = Y.shape[1]
  y = Y.shape[2]
  for i in range(0,s,skip + 1):
    groundTruth = np.fliplr(np.flipud(Y[i,:,:,0]))
    prediction  = np.fliplr(np.flipud(P[i,:,:,0]))
    diff        = groundTruth - prediction
    
    name = save_name_pref + str(number_zeroPadding(i)) + '_gT_' + '.png'
    matplotlib.image.imsave( os.path.join(out_path, name), groundTruth, cmap="gray")
    
    name = save_name_pref + str(number_zeroPadding(i)) + '_pred_' + '.png'
    matplotlib.image.imsave( os.path.join(out_path, name), prediction, cmap="gray")
    
    name = save_name_pref + str(number_zeroPadding(i)) + '_diff_' + '.png'
    matplotlib.image.imsave( os.path.join(out_path, name), diff, cmap="gray")
  
def batch_arr_to_itkImgs(B, ref_slice):
    s = B.shape[0]
    x = B.shape[1]
    y = B.shape[2]
    result = []
    for i in range(0,s):
        pred_arr        = np.zeros((x,y,1), dtype=np.float32)
        pred_arr[:,:,0] = B[i,:,:,0]
        
        prediction      = sitk.GetImageFromArray(pred_arr)         
        prediction.CopyInformation(ref_slice)
        result.append(prediction)
    
    return result

def get_groundTruth_prediction_pairs(Y, P, skip=0, out_path='', save_name_pref=''):  
    s = Y.shape[0]
    x = Y.shape[1]
    y = Y.shape[2]
    result = []
    for i in range(0,s,skip + 1):
        groundTruth = np.fliplr(np.flipud(Y[i,:,:,0]))
        prediction  = np.fliplr(np.flipud(P[i,:,:,0]))
        diff        = groundTruth - prediction

        temp = {'images':{},'names':{}}

        temp['images']['ground_truth'] = groundTruth 
        temp['images']['pred']         = prediction 
        temp['images']['diff']         = diff

        temp['names']['ground_truth'] = save_name_pref + str(number_zeroPadding(i)) + '_gT_' + '.png'
        temp['names']['pred']         = save_name_pref + str(number_zeroPadding(i)) + '_pred_' + '.png'
        temp['names']['diff']         = save_name_pref + str(number_zeroPadding(i)) + '_diff_' + '.png'

        result.append(temp)
    return result
    
def auto_predict_2D(model, generator,  batch_num=15, step=180, out_path='', save_name_pref='', save_name_suf=''):    
    for i in range(0, batch_num*step, step):
        refVols         = []
        caseIDs         = []
        sample_ids_temp = []
        (X_val, Y_val) = generator.__getitem__(index=i, inference='single_slice', return_RefVol=True, ref_vols=refVols, caseIDs=caseIDs, batch_IDs=sample_ids_temp)
        
        prediction = model.predict(X_val, batch_size=1)
        
        # visualize_sample(X_val, Y_val, prediction, out_path=out_path, save_name_pref=save_name_pref + '_' + caseIDs[0] + save_name_suf + '_in_out_{}_'.format(number_zeroPadding(i)))
        # save_groundTruth_prediction_pairs(Y_val, prediction, out_path=out_path, save_name_pref=save_name_pref + '_' + caseIDs[0] + save_name_suf + '_TRE_')
        
        # result_name = save_name_pref + '_' + caseIDs[0] + '_ref_3D_{}.nii.gz'.format(number_zeroPadding(i))
        # sitk.WriteImage(refVols[0], os.path.join(out_path, result_name))
        # print('ref vol size: ', refVols[0].GetSize())
        # print('ref vol size: ', refVols[0].GetWidth(),  refVols[0].GetHeight(), refVols[0].GetDepth())
        # ref_slice         = sitk.Extract(refVols[0], (0, refVols[0].GetHeight(), refVols[0].GetDepth()), (150, 0, 0))
        ref_slice         = refVols[0][150:151,:,:]
        # print('ref slice size: ',ref_slice.GetSize())
        prediction_itkImg = batch_arr_to_itkImgs(prediction, ref_slice)[0]
    
        
        # result_name = save_name_pref + '_' + caseIDs[0] + '_ref_2D_{}.nii.gz'.format(number_zeroPadding(i))
        # sitk.WriteImage(ref_slice, os.path.join(out_path, result_name))
        
        result_name = save_name_pref + '_' + caseIDs[0] + '_pred_2D_{}.nii.gz'.format(number_zeroPadding(i))
        sitk.WriteImage(prediction_itkImg, os.path.join(out_path, result_name))
    
        
        # res_img_arr = prediction[0,:,:,0]
        # pred = np.fliplr(np.flipud(res_img_arr))
        # pred = prep.normalize_numpy(pred) * 255
        # pred = pred.astype(int)
        
        # result_name = save_name_pref + str(i) + '_pred_2D.png'
        # matplotlib.image.imsave( os.path.join(out_path, result_name), pred, cmap="gray")
    
def get_prediction_2D(model, generator,  batch_num=15, step=180, out_path='', save_name_pref='', save_name_suf=''):    
    for i in range(0, batch_num*step, step):
        caseIDs = []
        (X_val, Y_val) = generator.__getitem__(index=i, inference='single_slice', caseIDs=caseIDs)
        
        prediction = model.predict(X_val, batch_size=1)
        
        return get_groundTruth_prediction_pairs(Y_val, prediction, out_path=out_path, save_name_pref=save_name_pref + '_' + caseIDs[0] + save_name_suf + '_TRE_')
            


def auto_predict_3D_from_2D(model, generator, batch_num=90, step=30, max_vols=10, max_slices=90, out_path='', save_name_pref=''):
   for i in range(batch_num):
        refVols = []
        caseIDs = []
        batchIDs = []
        tic1 = time.perf_counter()
        (X_val, Y_val) = generator.__getitem__(index=i*step, inference='3D_from_2D', return_RefVol=True, ref_vols=refVols, caseIDs=caseIDs, batch_IDs=batchIDs)
        toc1 = time.perf_counter()
        
        tic2 = time.perf_counter()
        prediction = model.predict(X_val)
        toc2 = time.perf_counter()
        print(f"{i} (batch construction, volume reconstruction) = ({toc1 - tic1:0.3f}s, {toc2 - tic2:0.3f}s)")
        
        
        ### put volume together
        volSize = refVols[0].GetSize()
        result = np.zeros((prediction.shape[1],prediction.shape[2],prediction.shape[0]), dtype=np.float32)
        for j in range(prediction.shape[0]):
            result[:,:,j] = prediction[j,:,:,0]

        if max_vols > 0:            
            # visualize_sample(X_val, Y_val, prediction, out_path=out_path, save_name_pref=save_name_pref + '_' + caseIDs[0] + '_in_out_{}_'.format(i))
            res_img = sitk.GetImageFromArray(result)
            res_img.CopyInformation(refVols[0])

            # result_name = save_name_pref + '_' + caseIDs[0] + '_pred_3D_From_2D_{}.nrrd'.format(i)
            result_name = save_name_pref + '_' + caseIDs[0] + '_pred_{}.nii.gz'.format(number_zeroPadding(i))
            sitk.WriteImage(res_img, os.path.join(out_path, result_name))
            max_vols -= 1
        
        
        if max_slices > 0:
            #result_crosssection = np.fliplr(np.flipud(result[:,29,:]))         
            result_crosssection = np.fliplr(np.flipud(result[:,58,:]))         
            file_name = save_name_pref + '_' + caseIDs[0] + '_crossSection_{}.png'.format(number_zeroPadding(i))
            matplotlib.image.imsave(os.path.join(out_path, file_name), result_crosssection, cmap="gray")
            
            # result_mip = np.fliplr(np.flipud(np.max(result[:,:,:], axis=1)))         
            # file_name = save_name_pref + '_' + caseIDs[0] + '_mip_{}.png'.format(i)
            # matplotlib.image.imsave(os.path.join(out_path, file_name), result_mip, cmap="gray")
            max_slices -= 1

def auto_evaluate(model, generator, batch_num=90, step=1, save_pref='', mask=False):
    mse    = []
    mae    = []
    rmse   = []
    mape   = []
    cosine = []
    mdisp  = []
    mmdisp = []
    drmse  = []
    
    computation_times = []
    sample_ids        = []
    series_numbers    = []
    series_timePoint  = []
 
    ### preper mask
    if mask:   
        refVols         = []
        caseIDs         = []
        sample_ids_temp = []
        slice_pos_px    = []
        (X_val, Y_val) = generator.__getitem__(index=1*step, inference='single_slice', return_RefVol=True, ref_vols=refVols, caseIDs=caseIDs, batch_IDs=sample_ids_temp, slice_pos_px=slice_pos_px)

        ### load mask
        data_dir = os.path.split(sample_ids_temp[0])[0]
        mask_vol_path = glob.glob(os.path.join(data_dir, '*_liver_mask_vol.nii'))[0]
        
        mask_vol      = sitk.ReadImage(mask_vol_path)
        mask_eroded   = sitk.BinaryErode(mask_vol, (5,5,0))
        
        ### crop mask to data size        
        ref_slice = refVols[0][slice_pos_px[0]:slice_pos_px[0]+1,:,:]
        label_itk = batch_arr_to_itkImgs(Y_val, ref_slice)[0] ### batch size is 1
        do_mm     = label_itk.GetOrigin()
        de_mm     = label_itk.TransformContinuousIndexToPhysicalPoint(label_itk.GetSize())
        do_px     = mask_eroded.TransformPhysicalPointToContinuousIndex(do_mm)
        de_px     = mask_eroded.TransformPhysicalPointToContinuousIndex(de_mm)
        
        mask_cropped =  mask_eroded[:, int(do_px[1]):int(de_px[1]), int(do_px[2]):int(de_px[2])]
        # vz = mask_cropped.GetSize()
        # print('cropped vol size in px: lr={}(x), ap={}(y), is={}(z)'.format(vz[0], vz[1], vz[2]))
        
    for i in tqdm(range(batch_num)):
        # print('batch {} / {}'.format(i, batch_num))
        refVols         = []
        caseIDs         = []
        sample_ids_temp = []
        slice_pos_px    = []
        
        start_time = time.time()
        (X_val, Y_val) = generator.__getitem__(index=i*step, inference='single_slice', return_RefVol=True, ref_vols=refVols, caseIDs=caseIDs, batch_IDs=sample_ids_temp, slice_pos_px=slice_pos_px)
        
        ### compute intensity losses

        loss_vals      = model.evaluate(X_val, Y_val,verbose=0)

        mse.append(   loss_vals[1])            
        mae.append(   loss_vals[2])    
        rmse.append(  np.sqrt(loss_vals[1]))   
        mape.append(  loss_vals[4])   
        cosine.append(loss_vals[5]) 
            
        ### compute local mean displacment
        prediction     = model.predict(X_val, batch_size=1)
        ref_slice      = refVols[0][slice_pos_px[0]:slice_pos_px[0]+1,:,:]
        prediction_itk = batch_arr_to_itkImgs(prediction, ref_slice)[0] ### batch size is 1
        label_itk      = batch_arr_to_itkImgs(Y_val,     ref_slice)[0] ### batch size is 1
        
        # result_name = caseIDs[0] + '_pred_2D_{}.nii.gz'.format(number_zeroPadding(i))
        # sitk.WriteImage(prediction_itk, os.path.join('/data/gino/4D_MRI_CNN/output/debug/mean_displacement_error', result_name))
        
        # result_name = caseIDs[0] + '_gt_2D_{}.nii.gz'.format(number_zeroPadding(i))
        # sitk.WriteImage(label_itk, os.path.join('/data/gino/4D_MRI_CNN/output/debug/mean_displacement_error', result_name))
        
        ### remove pseudo 3rd dimension
        label_2d = sitk.Extract(label_itk, (0, label_itk.GetHeight(), label_itk.GetDepth()), (0,0,0))
        pred_2d  = sitk.Extract(prediction_itk, (0, prediction_itk.GetHeight(), prediction_itk.GetDepth()), (0,0,0))
               
        mask_slice = None
        if mask:
            ### slice from mask        
            data_index = mask_cropped.TransformPhysicalPointToIndex(label_itk.GetOrigin())
            mask_slice = prep.cropSliceFromVolume_X(mask_cropped, data_index[0]) ### x = rl, y = ap, z = is

        transform     = img_p.registration_2D(label_2d, pred_2d, grid_size=[8,8])        

        ### calc masked mean displacement
        ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm, ls_dx_px, ls_dy_px, ls_ox_px, ls_oy_px = sample_displacements(label_2d, transform, sampling_grid=(16, 16), img_mask=mask_slice)
        displacements = cal_displacements(ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm)
        mean_displacement = np.mean(np.array(displacements))
        mmdisp.append(mean_displacement)
        
        ### calc mean displacement
        ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm, ls_dx_px, ls_dy_px, ls_ox_px, ls_oy_px = sample_displacements(label_2d, transform, sampling_grid=(16, 16), img_mask=None)
        displacements = cal_displacements(ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm)
        mean_displacement = np.mean(np.array(displacements))
        mdisp.append(mean_displacement)
        
        ### calc deformed rmse 
        ### compute deformed pred
        deformed_pred     = img_p.deform_img(label_2d, pred_2d, transform)
        deformed_pred_arr = sitk.GetArrayFromImage(deformed_pred)
        label_2d_arr      = sitk.GetArrayFromImage(label_2d)
        # pred_2d_arr       = sitk.GetArrayFromImage(pred_2d)
        # plt.imsave('/data/gino/4D_MRI_CNN/output/debug/mask_displacment_error/deformed_pred.png', deformed_pred_arr)
        # plt.imsave('/data/gino/4D_MRI_CNN/output/debug/mask_displacment_error/pred.png', pred_2d_arr)
        # plt.imsave('/data/gino/4D_MRI_CNN/output/debug/mask_displacment_error/label.png', label_2d_arr)
        deformed_mse  = np.mean(np.square(label_2d_arr - deformed_pred_arr))
        deformed_rmse = np.sqrt(deformed_mse)
        drmse.append(deformed_rmse)
        # print('mse = {}, deformed_mse = {} '.format(loss_vals[1], deformed_mse))
        # print('rmse = {}, deformed_rmse = {} '.format(np.sqrt(loss_vals[1]), deformed_rmse))
        
        # result_name = caseIDs[0] + '_predDeformed_2D_{}.nii.gz'.format(number_zeroPadding(i))
        # sitk.WriteImage(deformed_pred, os.path.join('/data/gino/4D_MRI_CNN/output/debug/mean_displacement_error', result_name))
        if save_pref != '':
            img_names = []
            img_names.append('{}_{}_viewOrder_{}_{}_gt_deformation_field.png'.format(   save_pref, caseIDs[0], number_zeroPadding(i), 2))
            img_names.append('{}_{}_viewOrder_{}_{}_pred_deformation_field.png'.format( save_pref, caseIDs[0], number_zeroPadding(i), 1))
            img_names.append('{}_{}_viewOrder_{}_{}_moved_deformation_field.png'.format(save_pref, caseIDs[0], number_zeroPadding(i), 3))
            draw_displacementField_on_imgs([label_2d, pred_2d, deformed_pred], transform, save_path='/data/gino/4D_MRI_CNN/output/debug/mask_displacment_error', img_names=img_names, img_mask=mask_slice)
        
        end_time = time.time()
        computation_times.append(end_time-start_time)
        series_numbers.append(getSeriesNumber_From_SampleID(sample_ids_temp[-1]))
        series_timePoint.append(getSeriesTimePoint_From_SampleID(sample_ids_temp[-1]))
        sample_ids.append(sample_ids_temp[-1])
    return mse, mae, rmse, drmse, mape, cosine, mdisp, mmdisp, computation_times, series_numbers, series_timePoint, sample_ids
        
def print_attantion(title='', text=''):
    print()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!  Attantion !! !!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Title: {}'.format(title))
    print(text)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print()
    
    
def getSeriesNumber_From_SampleID(fname):    
  fname = os.path.split(fname)[1] 
  seriesNumber = int(fname.split('_')[3])
  return seriesNumber
            
def getSeriesTimePoint_From_SampleID(fname):    
  fname = os.path.split(fname)[1] 
  seriesTP = int(fname.split('_')[5].split('.')[0])
  return seriesTP

def groupFileNamesBySeriesNumber(fnames):  
  fileCount = len(fnames)
  result = [] ### will be a list of lists
  result.append([])

  fname = os.path.split(fnames[0])[1] 
  setNumber_last = fname.split('_')[3]
  for i in range(fileCount):

    ### meta data
    fname = os.path.split(fnames[i])[1]
    setNumber = fname.split('_')[3]

    if setNumber == setNumber_last:
      result[-1].append(fnames[i])
    else:
      result.append([]) 
      result[-1].append(fnames[i])

    setNumber_last = setNumber
  return result

def groupFileNamesBySeriesNumber_dict(fnames):  
  fileCount = len(fnames)

  fname = os.path.split(fnames[0])[1] 
  setNumber_last = fname.split('_')[3]
  
  result = {int(setNumber_last):[]} ### will be a dict of lists
  for i in range(fileCount):

    ### get first series number 
    fname = os.path.split(fnames[i])[1]
    setNumber = fname.split('_')[3]

    if setNumber == setNumber_last:
      result[int(setNumber)].append(fnames[i])
    else:
      result[int(setNumber)] = [] 
      result[int(setNumber)].append(fnames[i])

    setNumber_last = setNumber
  return result

def number_zeroPadding(num):
    num = int(num)
    padding = ''
    if num < 1000:
        padding = '0'
    if num < 100:
        padding = '00'
    if num < 10:
        padding = '000'
    return padding + str(num)


def number_zeroPadding_3(num):
    num = int(num)
    padding = ''
    if num < 100:
        padding = '0'
    if num < 10:
        padding = '00'
    return padding + str(num)

### find model with ID
def find_model_with_ID(model_main_path='', ID=''):
    model_path  = ''
    json_path   = ''
    found       = []
    for dir, subdirs, files in os.walk(model_main_path):
        if dir.find(ID) >= 0:
            found.append(dir)
            # model_path = os.path.join(dir, 'model_final.h5')
            model_path = glob.glob(os.path.join(dir, '*_best.h5'))[0]        
            json_path  = glob.glob(os.path.join(dir, '*.json'))[0]

    if len(found) == 0:
        print_attantion(title='ID not found', text='could not find specified ID: {}'.format(ID))
        return model_path, json_path, -1
    if len(found) > 1:
        text = 'Specified ID is not unambiguous : {} \n'.format(ID) + 'Found ID in following locations:{}'.format(found)
        print_attantion(title='To many IDs found', text=text)
        return model_path, json_path, -2
    
    return model_path, json_path, 0
   
### concatenates several parts of a name with "_"
def concat_name(parts):
    result = str(parts[0])
    for p in parts[1:]:
        result = result + '_' + str(p)
    return result


def csv_to_json(csv_name, json_name):
    ranges = {} # nested dict structure
    last_case = ''
    with open(csv_name, 'r', newline='') as csvfile:    
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            case = row['case']
            if case != last_case:
                last_case = case
                ranges[case]                = {}
                ranges[case]['group']       = row['group']
                ranges[case]['seq_ranges']  = {}
            seq_name    = int(row['seq'])                
            start_index = int(row['start_index'])
            end_index   = int(row['end_index'])
            ranges[case]['seq_ranges'][seq_name] = [start_index, end_index]
    
    with open(json_name, 'w') as f:
        json.dump(ranges, f, indent=4)
        
# csv_to_json(mdna_ranges_csv, '/home/gino/projects/4D_MRI_CNN/experiments/mdna_ranges.json')

def get_ensemble_from_csv(ensemble_csv):
    ens = {}
    with open(ensemble_csv, 'r', newline='') as csvfile:    
        csv_reader = csv.DictReader(csvfile)
        last_target_case = ''
        for row in csv_reader:
            target_case = row['target_case']
            if target_case != last_target_case:
                ens[target_case]                   = {}
                ens[target_case]['split_train']    = row['split_train']
                ens[target_case]['ID-source_Case'] = {}
                last_target_case                   = target_case
            ens[target_case]['ID-source_Case'][row['model_ID']]  = row['source_case']
    return ens

def cm_to_inch(value):
    return value/2.54

def draw_displacementField_on_imgs(imgs, transformation, save_path, img_names, sampling_grid = (18, 22), img_mask=None):
    if type(imgs) != list:
        imgs = [imgs]
        
    if type(img_names) != list:
        img_names = [img_names]
        
    my_red_cmap = plt.cm.Reds
    my_red_cmap.set_under(color="white", alpha="0")
    
    fig, ax = plt.subplots() 
    ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm, ls_dx_px, ls_dy_px, ls_ox_px, ls_oy_px = sample_displacements(imgs[0], transformation, sampling_grid, img_mask)
    
    displacements = []
    for dx_mm, dy_mm, ox_mm, oy_mm, dx_px, dy_px, ox_px, oy_px in zip(ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm, ls_dx_px, ls_dy_px, ls_ox_px, ls_oy_px):
        plt.plot([ox_px, dx_px], [oy_px, dy_px], linewidth=0.2, color='white')
        plt.scatter(ox_px, oy_px , s=2, linewidths=0.2,  facecolors='none', edgecolors='white' ) 
        dx = ox_mm-dx_mm
        dy = oy_mm-dy_mm
        dp = math.sqrt(dx*dx + dy*dy)
        displacements.append(dp)  
            
    ### draw the first image below  the arrwos
    img_obj = ax.imshow(sitk.GetArrayFromImage(imgs[0]), cmap='gray')
    ### draw mask on top
    # if img_mask != None:
        # img_mask_arr = sitk.GetArrayFromImage(img_mask)[:, :, 0]
        # img_obj = ax.imshow(img_mask_arr, cmap = my_red_cmap, alpha = 0.3)
    plt.savefig(os.path.join(save_path, img_names[0]), dpi=300) 
    
    ### update other images if available
    if len(imgs) > 1: 
        for img, name in zip(imgs[1:], img_names[1:]):
            img_obj.set_data(sitk.GetArrayFromImage(img))
            fig.canvas.flush_events()
            plt.savefig(os.path.join(save_path,name), dpi=300) 
            
    return displacements

def sample_displacements(img, transformation, sampling_grid = (18, 22), img_mask=None):
    ls_dx_mm = []
    ls_dy_mm = []
    ls_ox_mm = []
    ls_oy_mm = []
    ls_dx_px = []
    ls_dy_px = []
    ls_ox_px = []
    ls_oy_px = []
    
    for dx_px in np.linspace(0, img.GetWidth()-1, sampling_grid[0]):
        for dy_px in np.linspace(0, img.GetHeight()-1, sampling_grid[1]):
            if img_mask != None:
                img_mask_arr = sitk.GetArrayFromImage(img_mask)[:, :, 0]
                if img_mask_arr[int(dy_px), int(dx_px)] == 0: 
                    ### mask source points
                    continue
            ### sample points over whole img   
            dx_mm, dy_mm = img.TransformContinuousIndexToPhysicalPoint((dx_px, dy_px))  ### originX_physical ... 
            ox_mm, oy_mm = transformation.TransformPoint((dx_mm, dy_mm))
            ox_px, oy_px = img.TransformPhysicalPointToIndex((ox_mm, oy_mm))
            
            ls_dx_mm.append(dx_mm)
            ls_dy_mm.append(dy_mm)
            ls_ox_mm.append(ox_mm)
            ls_oy_mm.append(oy_mm)
            
            ls_dx_px.append(dx_px)
            ls_dy_px.append(dy_px)
            ls_ox_px.append(ox_px)
            ls_oy_px.append(oy_px)
            
    return [ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm, ls_dx_px, ls_dy_px, ls_ox_px, ls_oy_px]

def cal_displacements(ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm):
    displacements = []
    for dx_mm, dy_mm, ox_mm, oy_mm in zip(ls_dx_mm, ls_dy_mm, ls_ox_mm, ls_oy_mm):
        dx = ox_mm-dx_mm
        dy = oy_mm-dy_mm
        dp = math.sqrt(dx*dx + dy*dy)
        displacements.append(dp)  
    return displacements