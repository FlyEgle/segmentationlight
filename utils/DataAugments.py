"""
Implement some data transformers with opencv & numpy
@author: FlyEgle
@datetime: 2022-01-17
"""
import cv2 
import math
import torch 
import random 
import warnings
import numpy as np


class Scale:
    """Resize image with short edge or width==height image
    Args:
        scale_size: (int or list)
    Returns:
        scale_image: (ndarray) the scaled images
    """
    def __init__(self, scale_size):
        self.scale_size = scale_size
        self.isShort = False
        if isinstance(self.scale_size, int):
            self.isShort = True 
        elif isinstance(self.scale_size, tuple):
            if len(self.scale_size)  == 1:
                self.isShort = True 
            
    def _resize_short(self, images, targets):
        height, width, _ = images.shape
        min_edge = min(height, width)
        ratio = self.scale_size / min_edge
        scale_h = round(height * ratio)
        scale_w = round(width * ratio)
        scale_images = cv2.resize(images, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
        scale_targets =  cv2.resize(targets, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)
        
        return scale_images, scale_targets
    

    def _resize_equal(self, images, targets):
        scale_images = cv2.resize(images, (self.scale_size[0], self.scale_size[1]), interpolation=cv2.INTER_LINEAR)
        scale_targets = cv2.resize(targets, (self.scale_size[0], self.scale_size[1]), interpolation=cv2.INTER_NEAREST)

        return scale_images, scale_targets


    def __call__(self, imgs, tgts):
        if self.isShort:
            scale_images, scale_targets = self._resize_short(imgs, tgts)
        else:
            scale_images, scale_targets = self._resize_equal(imgs, tgts)
        
        return scale_images, scale_targets
        

class RandomCrop:
    def __init__(self, crop_size):
        """Random Crop the crop images with targets
        Args:
            crop_size : (int or list) crop size for image and targets
        Returns:
            crop_imgs : (ndarray) crop images or src images
            crop_tgts : (ndarray) crop targets or src targets, the shape with imgs
        """
        self.crop_size = crop_size.copy()
        if isinstance(self.crop_size, int):
            self.crop_size = [self.crop_size, self.crop_size]
        elif isinstance(self.crop_size, tuple):
            self.crop_size = list(self.crop_size)

        self.scale_size = crop_size
        self.scale_fun =  Scale(crop_size)

    def _isBG(self, tgts):
        """If the targets all is 0, 0 is background
        """
        if np.sum(tgts) == 0:
            return True 
        else:
            return False

    def __call__(self, imgs, tgts):
        height,  width, _ = imgs.shape
        # width
        if self.crop_size[0] >= width:
            self.crop_size[0] = width
        # height
        if self.crop_size[1] >= height:
            self.crop_size[1] = height
        
        # loop for random crop which crop include  all is bg
        for _ in range(10): 
            # random
            random_y = random.choice([y for y in range(height - self.crop_size[1] + 1)])
            random_x = random.choice([x for x in range(width - self.crop_size[0] +  1)])

            crop_imgs = imgs[random_y:random_y + self.crop_size[1], random_x: random_x + self.crop_size[0]]
            crop_tgts = tgts[random_y:random_y + self.crop_size[1], random_x: random_x + self.crop_size[0]]

            # scale for image small than crop size 
            new_h, new_w = crop_imgs.shape[0], crop_imgs.shape[1]
            if new_h != self.scale_size[0] or new_w != self.scale_size[1]:
                # scale 
                crop_imgs, crop_tgts = self.scale_fun(crop_imgs, crop_tgts)

            # only bg or forground
            if not self._isBG(crop_tgts):
                return crop_imgs, crop_tgts 

        return crop_imgs, crop_tgts


class CenterCrop:
    """center crop images, if image is smaller than crop size, pad 0 to crop images
    Args:
        crop_size:  (int or list) crop size, if int, cast  to list or tuple
    Returns:
        crop_imgs: crop images
        crop_tgts: crop targets, the shape is same with images
    """
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size

    def __call__(self, imgs, tgts):
        height, width, _ = imgs.shape 
        crop_w, crop_h = self.crop_size[0],  self.crop_size[1]
        if crop_w > width or crop_h > height:
            padding_ltrb = [
                (crop_w - width) // 2 if  crop_w > width else 0,
                (crop_h - height) // 2 if crop_h > height else 0,
                (crop_w - width + 1) // 2 if crop_w > width else 0,
                (crop_h - height + 1) // 2 if crop_h > height else 0,
            ]
            imgs = np.pad(imgs, ((padding_ltrb[1], padding_ltrb[3]), (padding_ltrb[0], padding_ltrb[2]), (0,0)), 'constant', constant_values=(0,0))
            if tgts.ndim == 3:
                tgts = np.pad(tgts, ((padding_ltrb[1], padding_ltrb[3]), (padding_ltrb[0], padding_ltrb[2]), (0,0)), 'constant', constant_values=(0,0))
            else:
                tgts = np.pad(tgts, ((padding_ltrb[1], padding_ltrb[3]), (padding_ltrb[0], padding_ltrb[2])), 'constant', constant_values=(0,0))
            height, width, _ = imgs.shape 
            if crop_w == width and crop_w == height:
                return imgs, tgts
        
        crop_y = int(round(height - crop_h)  / 2.)
        crop_x = int(round(width - crop_w) / 2.)


        return imgs[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w], tgts[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            

class RandomCropScale2:
    """RandomCrop with Scale the images & targets, if not crop fit size, need to switch the prob to do reisze to keep the over figure
        scale_size :  (list) a sequence of scale
        scale      :  default is (0.08, 1.0),  crop region areas
        ratio      :  default is (3. / 4., 4. / 3.), ratio for width / height
    Returns:
        scale_image : (ndarray) crop and scale image
        scale_target: (ndarray) crop and scale target, shape  is same with image
    """
    def __init__(self, scale_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), prob=0.5):
        self.scale_size = scale_size
        self.scale = scale 
        self.ratio = ratio   

        # self.prob = np.random.uniform(0, 1) > prob
        self.prob = prob
        self.scale_func = Scale(self.scale_size)

        # center crop
        # self.centercrop = CenterCrop(self.scale_size)

        if (self.scale[0] > self.scale[1]) or (self.ratio[0] >  self.ratio[1]):
            warnings.warn("Scale and ratio  should be of kind (min, max)")

    def _isBG(self, tgts):
        """If the targets all is 0, 0 is background
        """
        if np.sum(tgts) == 0:
            return True 
        else:
            return False

    # TODO: fix empty bug
    def _crop_imgs(self, imgs, tgts):
        height, width, _ = imgs.shape 
        area =  height * width 

        for _ in range(10):
            target_area = area * np.random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w < width and 0 < h < height:
                random_y = np.random.randint(0, height - h + 1)
                random_x = np.random.randint(0, width - w + 1)
                
                crop_image = imgs[random_y:random_y+h, random_x:random_x+w]
                crop_target = tgts[random_y:random_y+h, random_x:random_x+w]

                if not self._isBG(crop_target):
                    crop_image, crop_target = self.scale_func(crop_image, crop_target)
                    return crop_image, crop_target

            # switch prob or center crop
            if np.random.uniform(0, 1) > self.prob:
                # center crop
                in_ratio = float(width) / float(height)
                if in_ratio < min(self.ratio):
                    w = width
                    h = int(round(w / min(self.ratio)))
                elif in_ratio > max(self.ratio):
                    h = height
                    w = int(round(h * max(self.ratio)))
                else:
                    w = width
                    h = height 
                
                # navie center crop
                crop_x = max((width - w) // 2, 0)
                crop_y = max((height  - h) // 2, 0)
                imgs = imgs[crop_y:crop_y+height,  crop_x:crop_x+width]
                tgts = tgts[crop_y:crop_y+height, crop_x:crop_x+width]

            # scale 
            crop_image, crop_target = self.scale_func(imgs, tgts)
            return crop_image, crop_target


    def __call__(self, imgs, tgts):
        crop_image, crop_target = self._crop_imgs(imgs, tgts)
        return crop_image, crop_target


class RandomCropScale:
    """RandomCrop with Scale the images & targets, RandomCrop than Resize to scale size
    Args:
        scale_size :  (list) a sequence of scale
        scale      :  default is (0.08, 1.0),  crop region areas
        ratio      :  default is (3. / 4., 4. / 3.), ratio for width / height
    Returns:
        scale_image : (ndarray) crop and scale image
        scale_target: (ndarray) crop and scale target, shape  is same with image
    """
    def __init__(self, scale_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.scale_size = scale_size
        self.scale = scale 
        self.ratio = ratio   

        self.scale_func = Scale(self.scale_size)
        # center crop
        # self.centercrop = CenterCrop(self.scale_size)

        if (self.scale[0] > self.scale[1]) or (self.ratio[0] >  self.ratio[1]):
            warnings.warn("Scale and ratio  should be of kind (min, max)")

    def _isBG(self, tgts):
        """If the targets all is 0, 0 is background
        """
        if np.sum(tgts) == 0:
            return True 
        else:
            return False

    # TODO: fix empty bug
    def _crop_imgs(self, imgs, tgts):
        height, width, _ = imgs.shape 
        area =  height * width 

        for _ in range(10):
            target_area = area * np.random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w < width and 0 < h < height:
                random_y = np.random.randint(0, height - h + 1)
                random_x = np.random.randint(0, width - w + 1)
                
                crop_image = imgs[random_y:random_y+h, random_x:random_x+w]
                crop_target = tgts[random_y:random_y+h, random_x:random_x+w]

                if not self._isBG(crop_target):
                    return crop_image, crop_target

            # center crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:
                w = width
                h = height 
            
            # navie center crop
            crop_x = max((width - w) // 2, 0)
            crop_y = max((height  - h) // 2, 0)
            crop_image = imgs[crop_y:crop_y+height,  crop_x:crop_x+width]
            crop_target = tgts[crop_y:crop_y+height, crop_x:crop_x+width]

            return crop_image, crop_target


    def __call__(self, imgs, tgts):
        crop_image, crop_target = self._crop_imgs(imgs, tgts)
        scale_image, scale_target = self.scale_func(crop_image, crop_target)
        return scale_image, scale_target

                
class RandomHorizionFlip:
    """Random Vertical flip
    Args:
        p : (float) probility for random horizion flip
    Returns:
        flip_imgs or imgs
        flip_tgts or tgts
    """
    def __init__(self, p):
        if not isinstance(p, float):
            warnings.warn("p must be float")
        self.p = p 

    def _random_sampler(self, p):
        if np.random.uniform(0,1) >= p:
            return True 
        else:
            return False

    def __call__(self, imgs, tgts):
        # flip
        if self._random_sampler(self.p):
            flip_imgs = cv2.flip(imgs, 1)
            flip_tgts = cv2.flip(tgts, 1)
            return flip_imgs, flip_tgts
        else:
            return imgs, tgts


class RandomGaussianBlur:
    """RandomGaussianBlur
    """
    def __init__(self, p, kernel_size=(5, 5)):
        self.p = p
        self.kernel_size = kernel_size 
    
    def _random_sampler(self, p):
        if np.random.uniform(0,1) <= p:
            return True 
        else:
            return False

    def __call__(self, imgs, tgts):
        if self._random_sampler(self.p):
            imgs = cv2.GaussianBlur(imgs, self.kernel_size, 0)
        return imgs, tgts 


class RandomVerticalFlip:
    """Random Vertical flip
    Args:
        p : (float) probility for random vertical flip
    Returns:
        flip_imgs or imgs
        flip_tgts or tgts
    """
    def __init__(self, p):
        if not isinstance(p, float):
            warnings.warn("p must be float")
        self.p = p 

    def _random_sampler(self, p):
        if np.random.uniform(0,1) >= p:
            return True 
        else:
            return False

    def __call__(self, imgs, tgts):
        # flip
        if self._random_sampler(self.p):
            flip_imgs = cv2.flip(imgs, 0)
            flip_tgts = cv2.flip(tgts, 0)
            return flip_imgs, flip_tgts
        else:
            return imgs, tgts


# distort color 
def distort_color(img, prob=0.4):
    """distort color, inlcude brightness, contrast and color 
    Args:
        img  : a ndarray shape is (h, w, 3), value in [0-255]
        prob : a probility for random base
    Returns:
        img  : distort color image,  shape is (h, w, 3), value in [0-255]
    """
    def random_brightness(img, lower=prob, upper=1+prob):
        img = np.clip(img, 0.0, 1.0)
        e = np.random.uniform(lower, upper)
        # zero = np.zeros([1] * len(img.shape), dtype=img.dtype)
        return img * e # + zero * (1.0 - e)

    def random_contrast(img, lower=prob, upper=1+prob):
        e = np.random.uniform(lower, upper)
        gray = np.mean(img[:,:,0]) * 0.299 + np.mean(img[:,:,1]) * 0.587 + np.mean(img[:,:,2]) * 0.114
        mean = np.ones([1] * len(img.shape), dtype=img.dtype) * gray
        return img * e + mean * (1.0 - e)

    def random_color(img, lower=prob, upper=1+prob):
        e = np.random.uniform(lower, upper)
        gray = img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114
        gray = np.expand_dims(gray, axis=-1)
        return img * e + gray * (1.0 - e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    if img.dtype == 'uint8' and img.shape[-1] == 3:
        img = img.astype('float32')
        img = img * 1./255

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    img = np.clip(img, 0.0, 1.0)
    img *= 255
    img = img.astype(np.uint8)

    return img


class RandomColorJitter:
    """Random ColorJitter, for brightness, color and constant
    Args:
        prob : a float value for random seed
    Returns:
        imgs: after colorjitter imgs
        tgts: imgs mask 
    """
    def __init__(self, p):
        if np.random.uniform(0,1) >= p: 
            self.prob = 0.4
        else:
            self.prob = None 
    
    def __call__(self, imgs, tgts):
        assert imgs.shape[2] == 3, "inputs must be RGB"
        if self.prob is not None:
            imgs = distort_color(imgs, self.prob)
        
        return imgs, tgts


def isBG(images):
    if np.sum(images) == 0:
        return True 
    else:
        return False


def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class RandomRotate:
    """Random Rotate with -degree to degree, have two mode
        mode 1: do not change the image scale, so some obj may be occurate
        mode 2: change the image scale to make all obj in the image, if use need to scale
    Args:
        degree: 0-180 degree
        mode  : 0 or 1 for different mode
    Returns:
        rotate_imgs: (ndarray) rotate images, shape is same with src images
        rotate_tgts: (ndarray) rotate targets, shape is same with src targets 
    """
    def __init__(self, degree, mode):
        if isinstance(degree, int) or isinstance(degree, float):
            degree = [-degree, degree]
        elif isinstance(degree, tuple):
            degree = list(degree)

        self.degree = degree
        self.mode = mode 

    def __call__(self,  imgs,  tgts):
        # rotate with not expand the image 
        angle = np.random.randint(self.degree[0], self.degree[1])
        if self.mode == 0:
            for _ in range(10):
                rotate_imgs = rotate(imgs, angle)
                rotate_tgts = rotate(tgts, angle)
                if not isBG(rotate_tgts):
                    return rotate_imgs, rotate_tgts 
                else:
                    angle = np.random.randint(self.degree[0], self.degree[1])
            
        elif self.mode == 1:
            height, width = imgs.shape[:2]
            scale_func = Scale(scale_size=(width, height))
            for _ in range(10):
                rotate_imgs = rotate_bound(imgs, angle)
                rotate_tgts = rotate_bound(tgts, angle)
                
                rotate_imgs, rotate_tgts = scale_func(rotate_imgs, rotate_tgts)
                if not isBG(rotate_tgts):
                    return rotate_imgs, rotate_tgts 
                else:
                    angle = np.random.randint(self.degree[0], self.degree[1])
    
        return rotate_imgs, rotate_tgts


class RandomDegree:
    """Random rotate 90, 180, 270
    Returns:
        images: rotate images
        targets: rotate targets
    """
    def __init__(self):
        self.degree = [90, 180, 270]

    def __call__(self, images, targets):
        rotate = random.choice(self.degree)
        if rotate == 90:
            images = np.rot90(images, 1)
            targets = np.rot90(targets, 1)
        elif rotate ==  180:
            images = np.rot90(images, 2)
            targets = np.rot90(targets, 2)
        else:
            images = np.rot90(images, 3)
            targets = np.rot90(targets, 3)

        return images, targets


class Normalize:
    """Normalize with mean & std, default is the imagenet mean and std value
    Args:
        normalize : normalize with 0-1 by /255 
        mean      : default is [0.485, 0.456, 0.406] 
        std       : default is [0.229, 0.224, 0.225]
    Returns:
        imgs : Normalize images with /255  
        tgts : src tgts.
    """
    def __init__(self, normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if isinstance(mean, int):
            mean = [mean for _ in range(3)]
        elif  isinstance(mean, tuple):
            mean = list(mean)

        if isinstance(std, int):
            std = [std for _ in range(3)]
        elif  isinstance(std, tuple):
            std = list(std)

        self.NORM = normalize  # (0,1)
        self.mean = mean 
        self.std = std 
        
    def __call__(self, imgs, tgts):
        if self.NORM:
            imgs = imgs / 255.
        
        if self.std is not None:
            for i in range(imgs.shape[-1]):
                imgs[:,:,i] = (imgs[:,:,i] - self.mean[i]) / self.std[i]
        else:
            for i in range(imgs.shape[-1]):
                imgs[:,:,i] = imgs[:,:,i] - self.mean[i]
        
        return imgs, tgts


class ToTensor:
    """Translate the  ndarray to torch.Tensor, (h, w, c)->(c, h, w)
    Args:
        channel_first: (bool) (c h w) or (h w c)
    Returns:
        imgs_tensor: (torch.Tensor) (c h w) float tensor if channel_first is True
        tgts_tensor: (torch.Tensor) (h w) long tensor
    """
    def __init__(self, channel_first=True):
        self.channel_first = channel_first

    def __call__(self, imgs, tgts):
        if not isinstance(imgs, np.ndarray) and not isinstance(tgts, np.ndarray):
            raise TypeError("inputs type must be ndarray")

        if self.channel_first:
            imgs = np.transpose(imgs, (2,0,1))

        if tgts.ndim == 3:
            tgts = tgts[:,:,0]
            
        imgs_tensor = torch.from_numpy(imgs).float()
        tgts_tensor = torch.from_numpy(tgts).long()

        return imgs_tensor, tgts_tensor


class Compose:
    def __init__(self, func_list):
        if not isinstance(func_list, list) and not isinstance(func_list, tuple):
            raise TypeError("inputs must be list or tuple")

        self.func_list = func_list

    def __call__(self, imgs, tgts):
        for func in self.func_list:
            if func is not None:
                imgs, tgts = func(imgs, tgts)
        return imgs, tgts


class RanomCopyPastePruneBG(object):
    """this aug is only used for prune bg with cloth fg"""
    def __init__(self, prob=0.1):
        self.prob = prob 
        self.kernel_size = (5, 5)
        self.kernel = np.ones(self.kernel_size, np.uint8)
        self.bg_color = [0, 128, 255]

    def __call__(self, imgs, tgts):
        if np.random.uniform(0,1) <= self.prob:
            iterations = random.randint(0, 15)
            mask = tgts.copy()
            mask_dilation = cv2.dilate(mask, self.kernel, iterations)
            if np.any(mask_dilation==255):
                mask_dilation = mask_dilation / 255.
            points = np.where(mask_dilation==1)
            crop_imgs = imgs * mask_dilation
            bg_color = random.choice(self.bg_color)
            new_images = np.ones_like(imgs) * bg_color
            new_images[points] = crop_imgs[points]
            return new_images, tgts
        else:
            return imgs, tgts

        
if __name__ == '__main__':
    image_path = "/data/jiangmingchao/data/dataset/taobao_live_product/train_dataset_part6/image/044162/3.jpg"
    label_path = "/data/jiangmingchao/data/dataset/green_clothes_seg/Mask/taobao_rmbg_new14k/train_dataset_part6_image_044162_3.png"

    data_file = "/data/jiangmingchao/data/code/U-2-Net/makeData/taobao_green/taobao_seg_14k_green_tryon_1k_shuf.txt"

    import cv2 
    import json 
    from tqdm import tqdm 

    data_list = [json.loads(x.strip()) for x in open(data_file).readlines()][:10]
    copy_paste = RanomCopyPastePruneBG(0.5)
    count = 0 
    for data in data_list:
        image = cv2.imread(data["image_path"])
        label = cv2.imread(data["label_path"])

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        crop_image, mask = copy_paste(image, label)
        cv2.imwrite(f"/data/jiangmingchao/data/code/SegmentationLight/tmp/aug/{count}.jpg", crop_image)
        cv2.imwrite(f"/data/jiangmingchao/data/code/SegmentationLight/tmp/aug/{count}.png", label)
        count += 1
    # Crop = RandomCropScale(scale_size=(800, 800), scale=(0.5, 1))
    # for data in tqdm(data_list):
    #     if "tryon" in data["image_path"]:
    #         image = cv2.imread(data["image_path"])
    #         label = cv2.imread(data["label_path"])
    #         for idx in range(10):
    #             crop_img, crop_lbl = Crop(image, label)
    #             cv2.imwrite(f"./crop_image{idx}.png", crop_img)
    #         break

        # if crop_lbl.shape[0] != 800 and crop_lbl.shape[1] != 800:
        #     print(data)