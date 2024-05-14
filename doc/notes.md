

Dicom -> Nifti (data for deep learning)
==============================================================================
1. dicom2nii.py (transforms dicom into nifti and esteblishes the timestamp order)
1. dicomVol2nii.py (transforms star vibes into nifti volumes)
2. rename_pairs.py (forms pairs of ordert nav data files and renames them)


How were the (interleaved) sequences actually recorded by the MRI?
==============================================================================
- Slice recorded before navigator slice:
  - navigator actually recorded first
  - then data nav alternatingly
  - navigator saved as 2nd 
  - data saved as 1st and then in alternation
- Slice recording after navigator slice:
  - data actually recorded first
  - then nav data in alternation
  - but data saved as 2nd
- the nav pur sequences are recorded in the correct order

Translated with DeepL.com (free version)


Timestamp
==============================================================================
- acquisition time in descrip tag of nifti is corresponding to the one extracted from dicom earlier
- both are in format : hhmmss.ms


virtualenv
==============================================================================
```
> virtualenv name --python=python
> source name/bin/activate
> pip3 install -r req.txt
```
