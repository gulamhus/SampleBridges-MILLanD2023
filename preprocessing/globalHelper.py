"""
@author: Gino Gulamhussene

@description: The script is a collection of globally useful helper functions.
"""


""" ================== 
script imports
=================== """
import re
import os
import sys
import shutil
import csv
import random
import time
import inspect
import logging
import datetime

import matplotlib.pyplot as plt
import numpy as np
import dicom
import configparser

sys.path.append(r'C:\Tools\myPythonTools')
import tools



""" ================== 
global helper functions
=================== """
### checks if the file is a dicom file and has the prefix, that the user defined
def dicomTest(filename, prefix=''):
  dicomFile = open(filename, mode='rb')
  if not filename.startswith(prefix) or not tools.has_DICOM_header(dicomFile):
    print(filename + " : File has no dicom header or not the right prefix.")
    return False
  else:
    return True

### checks if dicom files in interleaved sequence and reference sequence have same dimensions
def checkDicomConsistency(probandMetadata):
  is_consistent = True
  pM = probandMetadata  
  ### go to ref folder and open dicom
  refSubFolder = r'Nav_Pur_' + str(pM.navSequeseNumber)
  refFolder = os.path.join(pM.probandFolder, 'MR_Data', refSubFolder)
  os.chdir(refFolder)
  fnames = os.listdir('.')
  if not dicomTest(fnames[0]):
    print(fnames[0], ' is no dicom file. Exit program.')
    exit()
  dfile = dicom.read_file(fnames[0])
  voxelSize_x_ref = float(dfile.PixelSpacing[0])
  voxelSize_y_ref = float(dfile.PixelSpacing[1])
  voxelSize_z_ref = float(dfile.SliceThickness)
  img = dfile.pixel_array.astype(np.float32) 
  width_ref = img.shape[0]
  height_ref = img.shape[1]
  
  intSubFolder = r'Nav_+_Data'
  intFolder = os.path.join(pM.probandFolder, 'MR_Data', intSubFolder)
  os.chdir(intFolder)
  fnames = os.listdir('.')
  if not dicomTest(fnames[0]):
    print(fnames[0], ' is no dicom file. Exit program.')
    exit()
  dfile = dicom.read_file(fnames[0])
  voxelSize_x_int = float(dfile.PixelSpacing[0])
  voxelSize_y_int = float(dfile.PixelSpacing[1])
  voxelSize_z_int = float(dfile.SliceThickness)
  img = dfile.pixel_array.astype(np.float32) 
  width_int = img.shape[0]
  height_int = img.shape[1]

  if voxelSize_x_ref != voxelSize_x_int:
    print('voxelSize_x_ref {} is not equal to voxelSize_x_int {}'.format(voxelSize_x_ref, voxelSize_x_int))
    is_consistent = False
  if voxelSize_y_ref != voxelSize_y_int:
    print('voxelSize_y_ref {} is not equal to voxelSize_y_int {}'.format(voxelSize_y_ref, voxelSize_y_int))
    is_consistent = False
  if voxelSize_z_ref != voxelSize_z_int:
    print('voxelSize_z_ref {} is not equal to voxelSize_z_int {}'.format(voxelSize_z_ref, voxelSize_z_int))
    is_consistent = False
  if width_ref != width_int:
    print('width_ref {} is not equal to width_int {}'.format(width_ref, width_int))
    is_consistent = False
  if height_ref != height_int:
    print('height_ref {} is not equal to height_int {}'.format(height_ref, height_int))
    is_consistent = False

  return is_consistent

### checks if images in interleaved sequence and reference sequence have same dimensions
def checkImageConsistency(probandMetadata):
  is_consistent = True
  pM = probandMetadata  

  ### check if packed data is available 
  os.chdir(os.path.join(pM.probandFolder, 'MR_Data'))
  navPurFileName = 'Nav_Pur_' + str(pM.navSequeseNumber) + '.nii.gz'
  navDataFileName = 'Nav_+_Data.nii.gz'
  if not os.path.isfile(navPurFileName):
    print(navPurFileName, ' does not exist in.', os.path.join(pM.probandFolder, 'MR_Data'), ' Exiting progam.' )
  if not os.path.isfile(navDataFileName):
    print(navDataFileName, ' does not exist in.', os.path.join(pM.probandFolder, 'MR_Data'), ' Exiting progam.' )

  ### loading image data as float32 (some the openCV functions only accept float32)
  navPur_nii = nib.load(os.path.join(pM.probandFolder, 'MR_Data', navPurFileName))
  width_ref = navPur_nii.shape[1]
  height_ref = navPur_nii.shape[2]
  
  navData_nii = nib.load(os.path.join(pM.probandFolder, 'MR_Data',navDataFileName))
  width_int = navData_nii.shape[1]
  height_int = navData_nii.shape[2]

  if width_ref != width_int:
    print('width_ref {} is not equal to width_int {}'.format(width_ref, width_int))
    is_consistent = False
  if height_ref != height_int:
    print('height_ref {} is not equal to height_int {}'.format(height_ref, height_int))
    is_consistent = False

  return is_consistent

### copies the orientation information from one nifti to another
def copyOrientation(nifti_meta, nifti_target):
  ### get most of the meta info from the reference nifti 
  meta_header = nifti_meta.header.copy()

  ### pixel dimensions
  pixdim = meta_header['pixdim']
  new_pixdim = [pixdim[0], pixdim[3], pixdim[1], pixdim[2], pixdim[4], pixdim[5], pixdim[6], pixdim[7]]
  nifti_target.header['pixdim'] = new_pixdim
  nifti_target.header['regular'] = 'r'
  nifti_target.header['xyzt_units'] = 10

  ### orientation
  srow_x = meta_header['srow_x']
  srow_y = meta_header['srow_y']
  srow_z = meta_header['srow_z']
  new_srow_x = [srow_x[2], 0., 0., srow_x[3]]
  new_srow_y = [0., 0., srow_y[0], srow_y[3]]
  new_srow_z = [0., -srow_z[1], 0., srow_z[3]]
  nifti_target.header['sform_code'] = 1
  nifti_target.header['srow_x'] = new_srow_x
  nifti_target.header['srow_y'] = new_srow_y
  nifti_target.header['srow_z'] = new_srow_z
  return nifti_target

### get the r location of dicom file
def getRLocation(dicomFile):
  tag2 = getattr(dicomFile, 'ImagePositionPatient', None)
  tag3 = getattr(dicomFile, 'ImageOrientationPatient', None)
  xdir = tag3[:3]
  ydir = tag3[3:]
  zdir = np.cross(xdir, ydir)
  r_location = np.dot(tag2, zdir)
  return r_location


### get the S - I location of dicom file
def getSLocation(dicomFile):
  tag2 = getattr(dicomFile, 'ImagePositionPatient', None)
  tag3 = getattr(dicomFile, 'ImageOrientationPatient', None)
  xdir = tag3[:3]
  ydir = tag3[3:]
  s_location = np.dot(tag2, ydir)
  return s_location

### get the A - P location of dicom file
def getALocation(dicomFile):
  tag2 = getattr(dicomFile, 'ImagePositionPatient', None)
  tag3 = getattr(dicomFile, 'ImageOrientationPatient', None)
  xdir = tag3[:3]
  a_location = np.dot(tag2, xdir)
  return a_location

currendRoi = 1
def key_press(event):
  if int(event.key) in range(1,10):
      print('you pressed', event.key, event.xdata, event.ydata)
      print(f'Set current ROI to {int(event.key):d}')
      global currendRoi
      currendRoi = int(event.key)

def configName(probandFolderLocal, navSequenceNumber):
  probandConfig = ""
  if navSequenceNumber > 1:    
    probandConfig = probandFolderLocal + '_' + str(navSequenceNumber) + '.cnf' 
  else:
    probandConfig = probandFolderLocal + '.cnf' 

  return probandConfig



def resliceImg(imgData, voxelSize, plains, axis, outputWidth=0):  
  voxelsize_x = voxelSize[0]
  voxelsize_y = voxelSize[1]
  voxelsize_z = voxelSize[2]
  x = plains[0]
  y = plains[1]
  z = plains[2]

  if axis == 0:
    ### reslice sagitalSlice
    imgSlice = imgData[x,:,:]
  elif axis == 1:
    ### reslice axialSlice
    imgSlice = np.rot90(imgData[:,y,:], 3)
    aspect = (imgSlice.shape[1] * voxelsize_z) / (imgSlice.shape[0] * voxelsize_x)
    newHeight = imgSlice.shape[0]
    newWidth = int(imgSlice.shape[0] * aspect)
    imgSlice = cv2.resize(imgSlice, dsize=(newWidth, newHeight), interpolation=cv2.INTER_NEAREST)
  elif axis == 2:
    ### reslice coronalSlice
    imgSlice = np.rot90(imgData[:,:,z], 3)
    aspect = (imgSlice.shape[1] * voxelsize_z) / (imgSlice.shape[0] * voxelsize_x)
    newHeight = imgSlice.shape[0]
    newWidth = int(imgSlice.shape[0] * aspect)
    imgSlice = cv2.resize(imgSlice, dsize=(newWidth, newHeight), interpolation=cv2.INTER_NEAREST)

  if outputWidth != 0:
    aspect = imgSlice.shape[0] / imgSlice.shape[1]
    newHeight = int(outputWidth * aspect)
    imgSlice = cv2.resize(imgSlice, dsize=(outputWidth, newHeight), interpolation=cv2.INTER_NEAREST)

  return imgSlice

def maskVolumesMutually(imgData1, imgData2):    
  blackSlicesPos1 = np.where(np.mean(imgData1, (1,2)) == 0)
  blackSlicesPos2 = np.where(np.mean(imgData2, (1,2)) == 0)
  blackSlicesPos = set(blackSlicesPos1[0]).union(set(blackSlicesPos2[0]))
  for p in blackSlicesPos:
    imgData1[p, :, :] = 0
    imgData2[p, :, :] = 0
  return (imgData1, imgData2)


def resliceAndCombineImg(imgData, voxelSize, plains, outputWidth=0):  
  voxelsize_x = voxelSize[0]
  voxelsize_y = voxelSize[1]
  voxelsize_z = voxelSize[2]
  x = plains[0]
  y = plains[1]
  z = plains[2]
  ### reslice sagitalSlice
  sagitalSlice = imgData[x,:,:]

  ### reslice axialSlice
  axialSlice = np.rot90(imgData[:,y,:], 3)
  aspect = (axialSlice.shape[1] * voxelsize_z) / (axialSlice.shape[0] * voxelsize_x)
  newHeight = axialSlice.shape[0]
  newWidth = int(axialSlice.shape[0] * aspect)
  axialSlice = cv2.resize(axialSlice, dsize=(newWidth, newHeight), interpolation=cv2.INTER_NEAREST)

  ### reslice coronalSlice
  coronalSlice = np.rot90(imgData[:,:,z], 3)
  aspect = (coronalSlice.shape[1] * voxelsize_z) / (coronalSlice.shape[0] * voxelsize_x)
  newHeight = coronalSlice.shape[0]
  newWidth = int(coronalSlice.shape[0] * aspect)
  coronalSlice = cv2.resize(coronalSlice, dsize=(newWidth, newHeight), interpolation=cv2.INTER_NEAREST)

  ### put it all in one img
  allInOneWidth = sagitalSlice.shape[1] + axialSlice.shape[1] + coronalSlice.shape[1] 
  allInOneHeight = sagitalSlice.shape[0] 
  allInOneImg = np.full((allInOneHeight, allInOneWidth), 0.0)

  ### axialSlice
  xstart = 0
  xend = xstart + axialSlice.shape[1]
  allInOneImg[0:axialSlice.shape[0], 0:axialSlice.shape[1]] = axialSlice

  ### sagitalSlice
  xstart = xend
  xend = xstart + sagitalSlice.shape[1]
  allInOneImg[0:sagitalSlice.shape[0], xstart:xend] = sagitalSlice

  ### coronalSlice
  xstart = xend
  xend = xstart + coronalSlice.shape[1]
  allInOneImg[0:coronalSlice.shape[0], xstart:xend] = coronalSlice

  if outputWidth != 0:
    aspect = allInOneImg.shape[0] / allInOneImg.shape[1]
    newHeight = int(outputWidth * aspect)
    allInOneImg = cv2.resize(allInOneImg, dsize=(outputWidth, newHeight), interpolation=cv2.INTER_NEAREST)

  return allInOneImg


def giveConfigsWithmutualSameReconstructionrate(configFolder_base, configFolder_adapted, minReconstructionRate):
  
  ### proband configs folders
  fileNames_adapted = os.listdir(configFolder_adapted)
  fileNames_base = os.listdir(configFolder_base)

  ### check if configs for both versions are consitent with respect to each other
  for i in range(len(fileNames_base)):
    if fileNames_base[i] != fileNames_adapted[i]:
      print('Config names not matching: ', fileNames_base[i], 'and', fileNames_adapted[i])
      exit()


  ### using base proband configs as reference
  ### filter list, e.g. for none available data
  print('==========================================================================')
  print('filtering out unavailable probands for base version')
  print('==========================================================================')
  goodConfigsAdapted = []
  goodConfigsBase = []
  goodConfigsBoth = []

  configNames_base = []
  for i in range(len(fileNames_base)):
    pM_adapted = probandMetaData()
    pM_base = probandMetaData()
    f = fileNames_base[i]
    if f.endswith(".cnf"):
      success = pM_base.readProbandConfig(configFolder_base, f)
      success = success and pM_adapted.readProbandConfig(configFolder_adapted, f)

      if success:
        ### check if parameters of configs do match
        check = True
        check = check and pM_adapted.navDataSequensesCount == pM_base.navDataSequensesCount
        #check = check and len(pM_adapted.timePoints) == len(pM_base.timePoints)
        check = check and pM_adapted.navRefSliceIndex == pM_base.navRefSliceIndex
        check = check and pM_adapted.navRLocation == pM_base.navRLocation
        check = check and pM_adapted.navSequeseNumber == pM_base.navSequeseNumber
        check = check and pM_adapted.simMeth == pM_base.simMeth
        check = check and pM_adapted.maxDisSimilarityPerRoi == pM_base.maxDisSimilarityPerRoi
        for j in pM_adapted.rois:
          check = check and pM_adapted.rois[j].get_x() == pM_base.rois[j].get_x()
          check = check and pM_adapted.rois[j].get_y() == pM_base.rois[j].get_y()
          check = check and pM_adapted.rois[j].get_width() == pM_base.rois[j].get_width()
          check = check and pM_adapted.rois[j].get_height() == pM_base.rois[j].get_height()
        if not check:
          print('Configs not matching: ', fileNames_base[i], 'and', fileNames_adapted[i])
          exit()
          
        if len(pM_adapted.timePoints) > 0:  
          tpCandidats_both = []
          tpCandidats_base = []
          tpCandidats_adapted = []
          for t in pM_base.timePoints:
            inBase = False
            inAdapted = False
            inBase    =    pM_base.timePoints[t][0] <=    (pM_base.navDataSequensesCount * (1.0 - minReconstructionRate))
            inAdapted = pM_adapted.timePoints[t][0] <= (pM_adapted.navDataSequensesCount * (1.0 - minReconstructionRate))
            if inAdapted and inBase:
              tpCandidats_both.append(t)
            if inBase:
              tpCandidats_base.append(t)
            if inAdapted:
              tpCandidats_adapted.append(t)

          if len(tpCandidats_both) > 0:
            goodConfigsBoth.append([f, tpCandidats_both])
            print(r'using config: ' + f)
          else: 
            print(r'Reconstruction rate for one of the configs < ', minReconstructionRate, ', skipping config: ' + f)
          if len(tpCandidats_base) > 0:
            goodConfigsBase.append([f, tpCandidats_base])
          if len(tpCandidats_adapted) > 0:
            goodConfigsAdapted.append([f, tpCandidats_adapted])

         
        else: 
          print(r'No timepoints in config, skipping config: ' + f)
      else: 
        print(r'Could not load, skipping config: ' + f)
  return (goodConfigsBoth, goodConfigsBase, goodConfigsAdapted) 

def generateFTXYZ(pM, goodTimePoints):
  ind = random.randint(0, len(goodTimePoints)-1)
  t = goodTimePoints[ind]

  ### x locations from wich x is coosen
  xCandidats1 =  np.where(np.array(pM.timePoints[t][2:]) > 0)[0]
  xCandidats2 = np.arange(5, int(pM.navRLocation/pM.voxelSize_z + 5), 1)
  xCandidats = np.intersect1d(xCandidats1, xCandidats2)
  ind = random.randint(0, len(xCandidats)-1)
  x = xCandidats[ind] 

  ### y  locations from wich y is coosen
  minY = pM.rois[1].get_y()
  maxY = minY + pM.rois[1].get_height()
  for r in pM.rois:
    newMinY = pM.rois[r].get_y()
    if newMinY < minY:
      minY = newMinY

    newMaxY = pM.rois[r].get_y() + pM.rois[r].get_height()
    if newMaxY > maxY:
      maxY = newMaxY

  y = random.randint(int(minY),int(maxY))
  
  ### z  locations from wich z is coosen
  maxZ = pM.rois[1].get_x() + pM.rois[1].get_width()
  minZ = pM.rois[1].get_x()
  for r in pM.rois:
    if pM.rois[r].get_x() + pM.rois[r].get_width() > maxZ:
      maxZ = pM.rois[r].get_x() + pM.rois[r].get_width()
    if pM.rois[r].get_x() < minZ:
      minZ = pM.rois[r].get_x()

  z = random.randint(int(minZ), int(maxZ))

  result = [t,x,y,z]
  return result


# ======================================================================================================
# ======================================================================================================
# synchronization
# ======================================================================================================
# ======================================================================================================

# ===================================================
# finds the
# finds out wich ones are data frames and returns a list of those file names
# ===================================================
def sync_curves(curve1, curve2, maxExpactedDrift=2000):
  t1, y1 = curve1
  t2, y2 = curve2

  ### resample both curves to 1ms per data point for template matching
  (t1_resampled, y1_resampled) = resampleTimeCurve(t1, y1, 1)
  (t2_resampled, y2_resampled) = resampleTimeCurve(t2, y2, 1)

  y1_resampled = np.float32(y1_resampled.reshape(y1_resampled.shape[0], 1))
  y2_resampled = np.float32(y2_resampled.reshape(y2_resampled.shape[0], 1))

  ### cut search region
  ind_start = np.where(t1_resampled == t2_resampled[0])[0][0]
  if ind_start > maxExpactedDrift:
    ind_start = ind_start - maxExpactedDrift
  else:
    ind_start = 0

  ind_end = np.where(t1_resampled == t2_resampled[-1])[0][0]
  if ind_end < len(t1_resampled) - maxExpactedDrift:
    ind_end = ind_end + maxExpactedDrift
  else: 
    ind_end = len(t1_resampled)

  ### estimate time shift
  res = cv2.matchTemplate(y1_resampled[ind_start:ind_end], y2_resampled , eval('cv2.TM_CCOEFF'))
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  t_shift = t1_resampled[ind_start:ind_end][max_loc[1]] - t2_resampled[0]

  return (t1_resampled, y1_resampled, t2_resampled, y2_resampled, t_shift)



def resampleTimeCurve(t, y, newSampling = 1):
  interpolation = CubicSpline(t, y)
  start_t = t[0]
  end_t = t[-1]
  resamplingTimePoints = np.arange(start_t, end_t, 1)
  y_resampled = interpolation(resamplingTimePoints)
  return (resamplingTimePoints, y_resampled)


# ===================================================
# generate mean positions for feature vectors
# ===================================================
def meanPos_from_surrogate(featureVectors):
  meanPositions = []
  for f in featureVectors:  
    x_temp = []
    y_temp = []
    z_temp = []
    for i in range(3, len(f), 3):
      x_temp.append(f[i-2])
      y_temp.append(f[i-1])
      z_temp.append(f[i])
  
    x = np.mean(np.array(x_temp))
    y = np.mean(np.array(y_temp))
    z = np.mean(np.array(z_temp))
    meanPositions.append([x, y, z])
  return meanPositions

# ===================================================
# generate a synch curve from the surrogate signal (feature vector signal)
# ===================================================
def get_sync_curve_from_surrogate(featureVectors):
  p1Z = []
  p2Z = []
  for i in range(len(featureVectors)):
    p1Z.append(featureVectors[i][2])
    p2Z.append(featureVectors[i][5])
  sync_curve = -(np.array(p1Z) + np.array(p2Z)) * 0.5
  return sync_curve

# ======================================================================================================
# ======================================================================================================
# MRI
# ======================================================================================================
# ======================================================================================================

# ===================================================
# go through dicom files in dicom_fnames, 
# finds out wich ones are data frames and returns a list of those file names
# ===================================================
def getDataFilesOfInterleavedSequences(dicom_fnames, navRLocation):  
  fileCount = len(dicom_fnames)
  result_dicom_fNames = []

  ### init progress bar und start stop watch
  tools.printProgressBar(0, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)
  startTime = time.time()

  onRightSide = True
  for i in range(fileCount):
    ### read dicom file
    # if not dicomTest(dicom_fnames[i]):
    #   print(dicom_fnames[i], ' is no dicom file. Exit program.')
    #   exit()
    dfile = dicom.read_file(dicom_fnames[i])
    
    ### meta data
    r_location = getRLocation(dfile)

    ### check wether we are still right of the nav r_location or already left of it
    if onRightSide:
      if i%2 == 0 and r_location == navRLocation: ### this is only true for the first slice on the left side
        onRightSide = False

    ### if right of nav r_location we must remove all even slices, because they are navigator slices
    if onRightSide:
      if i%2 == 0:
        result_dicom_fNames.append(dicom_fnames[i])
    
    ### if left of nav r_location we must remove all uneven slices, because they are navigator slices
    if not onRightSide:
      if i%2 == 1 :
        result_dicom_fNames.append(dicom_fnames[i])          

    tools.printProgressBar(i, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)

  endTime = time.time()
  print()
  print('Data packing took {} seconds.'.format(endTime - startTime))
  return result_dicom_fNames




# ===================================================
# go through dicom files in dicom_fnames, 
# finds out wich ones are nav frames and returns a list of those file names
# ===================================================
def getNavFilesOfInterleavedSequences(dicom_fnames, navRLocation):  
  fileCount = len(dicom_fnames)
  result_dicom_fNames = []

  ### init progress bar und start stop watch
  tools.printProgressBar(0, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)
  startTime = time.time()

  for i in range(fileCount):
    ### read dicom file
    # if not dicomTest(dicom_fnames[i]):
    #   print(dicom_fnames[i], ' is no dicom file. Exit program.')
    #   exit()
    dfile = dicom.read_file(dicom_fnames[i])
    
    ### meta data
    r_location = getRLocation(dfile)

    ### check for the nav r_location
    if r_location == navRLocation: 
      result_dicom_fNames.append(dicom_fnames[i])
    
    tools.printProgressBar(i, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)

  endTime = time.time()
  print()
  print('Data packing took {} seconds.'.format(endTime - startTime))
  return result_dicom_fNames



# ===================================================
# go through dicom files in dicom_fnames, 
# groupes the file names by series number
# ===================================================
def sortDataFilesBySeriesNumber(dicom_fnames):  
  fileCount = len(dicom_fnames)
  result_dicom_fNames = [] ### will be a list of lists
  result_dicom_fNames.append([])
  ### init progress bar und start stop watch
  tools.printProgressBar(0, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)
  startTime = time.time()

  setNumber_last = dicom_fnames[0].split('.')[3]
  for i in range(fileCount):

    ### meta data
    setNumber = dicom_fnames[i].split('.')[3]

    if setNumber == setNumber_last:
      result_dicom_fNames[-1].append(dicom_fnames[i])
    else:
      result_dicom_fNames.append([]) 
      result_dicom_fNames[-1].append(dicom_fnames[i])

    setNumber_last = setNumber

    tools.printProgressBar(i, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)

  endTime = time.time()
  print()
  print('Data packing took {} seconds.'.format(endTime - startTime))
  return result_dicom_fNames



# ===================================================
# define the template in dicom file
# returns x1, y1 the top left corner of template
#         x2, y2 the bottom right corner of template
# ===================================================
def defTemplatePos(dicom_image):
  fig, ax = plt.subplots()
  ax.set_aspect('equal', 'datalim')


  def line_select_callback(eclick, erelease):
      x1, y1 = eclick.xdata, eclick.ydata
      x2, y2 = erelease.xdata, erelease.ydata

      rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), alpha=0.5 )
      ax.add_patch(rect)
      x1 = rect.xy[0]
      y1 = rect.xy[1]
      x2 = x1 + rect.get_width()
      y2 = y1 + rect.get_height()

      print("x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2)

  rs = RectangleSelector(ax, line_select_callback,
                        drawtype='box', useblit=False, button=[1], 
                        minspanx=5, minspany=5, spancoords='pixels', 
                        interactive=True)

  imgplot = ax.imshow(dicom_image)
  imgplot.set_cmap(plt.gray())    
  plt.show()

# ===================================================
# parses the dicom tag AcquisitionTime to ms
# ===================================================
def ms_from_DicomTag_AcquisitionTime(t_dcm):
    [hms_s, ms_s] = t_dcm.split(".")
    h = float(hms_s[0:2])
    m = float(hms_s[2:4])
    s = float(hms_s[4:6])
    ms = float(ms_s[0:3])
    ms = ms + 1000*s + 60*1000*m + 60*60*1000*h
    return ms

# ===================================================
# reading the original MPT logs
# returns the sync curve between start line and end line
# ===================================================
def syncCurveFromMRI(directory, x1, x2, y1, y2):
  if not os.path.isdir(directory):
    print('directory does not exist: ', directory )
    exit()
  os.chdir(directory)
  fnames = os.listdir('.')

  ### load firtst image and cut template
  dicomFile = dicom.read_file(fnames[0])
  currentImg = dicomFile.pixel_array.astype(np.float32)
  template = currentImg[x1:x2, y1:y2]

  # show template 
  # fig,ax = plt.subplots()
  # plt.imshow(template, cmap='gray')
  # plt.show()

  dataT = []
  dataX = []
  dataY = []
  for i in range(1, len(fnames)):
    dicomFile = dicom.read_file(fnames[i])
    nextImg = dicomFile.pixel_array.astype(np.float32)
    method = eval('cv2.TM_CCOEFF_NORMED')

    ### get time stamp from dicom file
    t_dcm = dicomFile.AcquisitionTime  
    dataT.append(ms_from_DicomTag_AcquisitionTime(t_dcm))

    # Apply template Matching
    res= cv2.matchTemplate(nextImg,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    dataX.append(top_left[0])
    dataY.append(nextImg.shape[1]-top_left[1])
        
  return (dataT, dataX, dataY)


# ======================================================================================================
# ======================================================================================================
# Other Stuff
# ======================================================================================================
# ======================================================================================================

# ===================================================
# go through dicom files in dicom_fnames, extract the time stamps and return them as list
# ===================================================
def pack_timeStamps(dicom_fnames):
  ### init progress bar and start stop watch
  fileCount = len(dicom_fnames)
  tools.printProgressBar(0, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)
  startTime = time.time()

  ### go through all dicom files
  ### get time stamps
  timeStamps = []
  for i in range(fileCount):
    ### read dicom file
    # if not dicomTest(dicom_fnames[i]):
    #   print(dicom_fnames[i], ' is no dicom file. Exit program.')
    #   exit()
    dicomFile = dicom.read_file(dicom_fnames[i])

    ### collect time stamps
    t_dcm = dicomFile.AcquisitionTime  
    timeStamps.append(ms_from_DicomTag_AcquisitionTime(t_dcm))

    tools.printProgressBar(i, fileCount, prefix = 'Progress:', suffix = 'Complete', length = 50)

  endTime = time.time()
  print()
  print('Data packing took {} seconds.'.format(endTime - startTime))
  return timeStamps


# ===================================================
# from time point [y, m, d, h, min] to ordinal
# ===================================================
def timePointToOrdinal(y, m, d, h, minute):
  date_temp = datetime.date(y, m, d)
  ordinal = date_temp.toordinal()*10000 + h * 60 + minute
  return ordinal


def fromTimeOrdinalToShift(ordinal):
  m =  -0.04452456018364179 ### these linear parameters where obtained using clockDrift.py 
  n =  327931616.8286217    ### these linear parameters where obtained using clockDrift.py
  return m*ordinal + n

# ===================================================
# parses the proband config name to y, m, d
# ===================================================
def date_from_ProbandConfigName(name):
    [y, m, d] = name.split('_')[0].split('-')
    y = int(y)
    m = int(m)
    d = int(d)
    return [y, m, d]

# ===================================================
# create a function level logger
# ===================================================
def function_logger(name, file_level, console_level = None):
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




# ======================================================================================================
# ======================================================================================================
# Test the Stuff
# ======================================================================================================
# ======================================================================================================
# print(fromTimeOrdinalToShift(7368500609))