import numpy as np
import nrrd
import nibabel as nib
import os
import pickle
from cv2 import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def normalize(image, MIN_BOUND, MAX_BOUND):
    """
    This function normalizes CT image inside bounds MIN_BOUND, MAX_BOUND. 
    """
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def bound_box_and_reshape(img, slice_idx):
    """
    Crop given slice of image to lung size (remove redundant empty space) and reshape to 512x512 pxls. Return edited img_slice.
    """
    img_slice = img[:,:,slice_idx]
    rows = np.any(img_slice, axis=1)
    cols = np.any(img_slice, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    img_slice = resize(img_slice[rmin:rmax, cmin:cmax], (224,224))
    """
    plt.imshow(img_slice.squeeze())
    plt.show()
    """
    img_slice = np.transpose(img_slice[:, :, np.newaxis], axes = [2, 0, 1]).astype('float32')
    return img_slice


def mask_original(filepath_CT, filepath_mask):
    """
    Mask and normalize original CT. 
    """
    I = np.array(nib.load(filepath_CT).dataobj)
    I = normalize(I, -1350, 150)
    M, _ = nrrd.read(filepath_mask)
    
    nS = np.where(M==1, I, M)

    r = np.any(nS, axis=(1, 2))
    c = np.any(nS, axis=(0, 2))
    z = np.any(nS, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    nS = nS[rmin:rmax, cmin:cmax, zmin:zmax]

    zdiff = zmax - zmin 
    l = []
    step = zdiff//27
    for i in range (4,24):
        aa = nS[:,:,step*i] 
        l.append(aa)
        """
        #print(aa.shape)
        #plt.imshow(aa.squeeze(),)
        #plt.show()
        """
    selected_slices = np.stack(l, axis=2 )   
    return selected_slices



if __name__ == "__main__":
    dataset = "test"
    
    CTs_path = "../../data/images/" + dataset
    masks_path = "../../data/segmentations/" + dataset
    output_path = "../../data/processed/" + dataset + "-whole-sampled"
    
    for name_CT in os.listdir(CTs_path):
        if name_CT.endswith(".gz"):
            name_mask = name_CT.replace(".nii.gz", ".nrrd")

            filepath_CT = os.path.join(CTs_path, name_CT)
            filepath_mask = os.path.join(masks_path, name_mask)
            
            img = mask_original(filepath_CT, filepath_mask)
            
            n_slices = 20
            print(name_CT)
            for slice_idx in range(n_slices):
                try:
                    img_slice = bound_box_and_reshape(img, slice_idx)
                except: 
                    raise NameError('Bounding and reshaping failed.')

                slice_name = name_CT.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
                np.save(os.path.join(output_path, slice_name), img_slice)



    # For test generate all test files
    if(dataset == 'test'):
        info_file = "../../data/test.txt"
        f = open(info_file, "w+")
        for name_CT in os.listdir(CTs_path):
            if name_CT.endswith(".gz"):
                name_mask = name_CT.replace(".nii.gz", ".nrrd")

                filepath_CT = os.path.join(CTs_path, name_CT)
                filepath_mask = os.path.join(masks_path, name_mask)
                
                f.write(name_CT + ", ?\n")
                print(name_CT + ", ?\n")

        f.close()