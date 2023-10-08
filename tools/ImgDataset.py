import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random
import numpy as np
import random
def jointly_ablate_images(img1: Image.Image, img2: Image.Image, percentage: float,test_mode) -> (Image.Image, Image.Image):
    # Convert PIL images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    if test_mode == False:
        percentage = random.uniform(0, percentage)
    # Verify that the images have the same dimensions
    assert arr1.shape == arr2.shape, "The two images must have the same dimensions."

    # Stack arrays vertically
    combined = np.vstack((arr1, arr2))

    # Determine the number of pixels across both images
    total_pixels = combined.shape[0] * combined.shape[1]

    # Determine the number of pixels to retain
    num_retain = int(total_pixels * percentage)

    # Create an ablation mask with 10% ones and 90% zeros
    mask = np.concatenate((np.ones(num_retain), np.zeros(total_pixels - num_retain)))
    np.random.shuffle(mask)
    mask = mask.reshape(combined.shape[0], combined.shape[1], 1)  # add a third dimension
    mask = np.repeat(mask, 3, axis=2)  # repeat for RGB channels

    # Apply the mask
    ablated_combined = combined * mask.astype(combined.dtype)

    # Split the ablated_combined array back into individual image arrays
    split_idx = arr1.shape[0]
    ablated_arr1 = ablated_combined[:split_idx]
    ablated_arr2 = ablated_combined[split_idx:]

    # Convert ablated arrays back to PIL images
    ablated_img1 = Image.fromarray(ablated_arr1)
    ablated_img2 = Image.fromarray(ablated_arr2)

    return ablated_img1, ablated_img2
def randomly_ablate_train(img, ablation_ratio=0.2, ablation_value=0):
    """
    Randomly ablating a given percentage of pixels in the image.

    Parameters:
    - img: PIL image to be ablated
    - percentage: Fraction of pixels to be ablated (default: 0.9)
    - ablation_value: The value to set the ablated pixels to (default: 0)

    Returns:
    - Ablated PIL image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Generate a random mask for ablation
    percentage = random.uniform(0, ablation_ratio)
    mask = np.random.rand(*img_array.shape[:2]) < 1-percentage

    # Ablate the image
    if len(img_array.shape) == 3:  # if the image has multiple channels (e.g., RGB)
        for channel in range(img_array.shape[2]):
            img_array[mask, channel] = ablation_value
    else:
        img_array[mask] = ablation_value

    # Convert back to PIL Image and return
    return Image.fromarray(img_array)

def randomly_ablate_test(img, ablation_ratio=0.1, ablation_value=0):
    """
    Randomly ablating a given percentage of pixels in the image.

    Parameters:
    - img: PIL image to be ablated
    - percentage: Fraction of pixels to be ablated (default: 0.9)
    - ablation_value: The value to set the ablated pixels to (default: 0)

    Returns:
    - Ablated PIL image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Calculate the number of pixels to ablate
    total_pixels = img_array.shape[0] * img_array.shape[1]
    num_pixels_to_ablate = int((1-ablation_ratio)* total_pixels)

    # Generate random indices to ablate
    indices = np.random.choice(total_pixels, num_pixels_to_ablate, replace=False)
    rows = indices // img_array.shape[1]
    cols = indices % img_array.shape[1]

    # Ablate the image
    if len(img_array.shape) == 3:  # if the image has multiple channels (e.g., RGB)
        for channel in range(img_array.shape[2]):
            img_array[rows, cols, channel] = ablation_value
    else:
        img_array[rows, cols] = ablation_value

    # Convert back to PIL Image and return
    return Image.fromarray(img_array)
class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True,ablation_ratio = 0.1,ablation_ratio1 = 0.1,ablation_ratio2 = 0.1):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.ablation_ratio = ablation_ratio
        self.ablation_ratio1 = ablation_ratio1
        self.ablation_ratio2 = ablation_ratio2
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            path_to_search = r'.\{}\{}\{}'.format(parent_dir, self.classnames[i], set_)
            print(path_to_search)
            #all_files = sorted(glob.glob(parent_dir+'\\'+self.classnames[i]+'\\'+set_+'\\*shaded*.png'))
            #all_files = sorted(glob.glob(path_to_search, recursive=True))
            all_files = sorted(list_files(path_to_search))
            #all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('\\')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.test_mode == False:
                im = randomly_ablate_train(im, ablation_ratio=self.ablation_ratio, ablation_value=0)
            else:
                if i ==0:
                    im = randomly_ablate_test(im, self.ablation_ratio1, ablation_value=0)
                if i == 1:
                    im = randomly_ablate_test(im, self.ablation_ratio2, ablation_value=0)
            #im.show()
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])

class MultiviewImgDatasetBaseline(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True,ablation_ratio = 0.1):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.ablation_ratio = ablation_ratio

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            path_to_search = r'.\{}\{}\{}'.format(parent_dir, self.classnames[i], set_)
            print(path_to_search)
            #all_files = sorted(glob.glob(parent_dir+'\\'+self.classnames[i]+'\\'+set_+'\\*shaded*.png'))
            #all_files = sorted(glob.glob(path_to_search, recursive=True))
            all_files = sorted(list_files(path_to_search))
            #all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('\\')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead

        im1 = Image.open(self.filepaths[idx*self.num_views+1]).convert('RGB')
        im2 = Image.open(self.filepaths[idx*self.num_views+1]).convert('RGB')
        im1,im2 = jointly_ablate_images(im1,im2,percentage=self.ablation_ratio,test_mode=self.test_mode)
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return (class_id, torch.stack([im1,im2]), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])
    
def list_files(directory):
    with os.scandir(directory) as entries:
        return [entry.path for entry in entries if entry.is_file()]

class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, ablation_ratio = 0.1):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.ablation_ratio = ablation_ratio
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        
        for i in range(len(self.classnames)):
            #path_to_search = "C:\Users\dongs\Documents\mvcnn_pytorch-master\modelnet40_images_new-12x\modelnet40_images_new_12x\airplane\test\airplane_0627.obj.shaded_v001.png"
            path_to_search = r'.\{}\{}\{}'.format(parent_dir, self.classnames[i], set_)
            print(path_to_search)
            #all_files = sorted(glob.glob(parent_dir+'\\'+self.classnames[i]+'\\'+set_+'\\*shaded*.png'))
            #all_files = sorted(glob.glob(path_to_search, recursive=True))
            all_files = sorted(list_files(path_to_search))
            for file in all_files:
                file = '.\{}\{}\{}'.format(parent_dir, self.classnames[i], set_)+'\\'+file
            #print(all_files)
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('\\')[-3]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.test_mode == False:
            im = randomly_ablate_train(im, ablation_ratio=self.ablation_ratio, ablation_value=0)
        else:
            im = randomly_ablate_test(im, ablation_ratio=self.ablation_ratio, ablation_value=0)
            #im.show()
        if self.transform:
            im = self.transform(im)
        #print(im.shape)

        return (class_id, im, path)

