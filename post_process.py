import os 
import cv2 
import numpy as np 
from tqdm import tqdm 


def getShrink(mask, shrink=2):
    for _ in range(shrink):
        contours, hierachy = cv2.findContours(
            mask[:,:,0],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            -1,
            (125, 125, 125),
            1
        )

        mask[mask==np.array([125, 125, 125])] = 0

    return mask 


def edgePostProcess(mask, image):
    """Edge post Process
    Args:
        mask: a ndarray map, value is [0,255], shape is (h, w, 3)
        image: a ndarray map, value is 0-255, shape  is(h, w, 3)
    Returns:
        outputs: edge blur image
    """
    mask[mask==255] = 1
    mask = getShrink(mask)

    image = image * mask 
    image[image==0] = 255
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    new_mask = np.zeros(image.shape, np.uint8)
    contours, hierachy = cv2.findContours(
        mask[:,:,0],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(new_mask, contours, -1, (255, 255, 255), 5)
    output = np.where(new_mask==np.array([255, 255, 255]), blur_image, image)
    return output 

if True:
    # mask_folder = "/data/jiangmingchao/data/dataset/seg_exp/out_0.8_new_mask"
    # imge_folder = "/data/jiangmingchao/data/dataset/seg_exp/out_0.8_new_out"
    mask_folder = "/data/jiangmingchao/data/code/SegmentationLight/tmp/mask"
    imge_folder = "/data/jiangmingchao/data/code/SegmentationLight/tmp/out"

    mask_list = [os.path.join(mask_folder, x) for x in os.listdir(mask_folder)]
    imge_list = [os.path.join(imge_folder, x) for x in os.listdir(imge_folder)]

    print(len(mask_list))
    print(len(imge_list))

    output_folder = "/data/jiangmingchao/data/code/SegmentationLight/tmp/blur"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    for data in tqdm(zip(mask_list, imge_list)):
        mask_path, image_path = data[0], data[1]
        mask = cv2.imread(mask_path)
        cv2.imwrite("./pre.png", mask)
        # print(mask.shape)
        mask[mask==255] = 1
        mask = getShrink(mask)

        image = cv2.imread(image_path)
        image = image * mask 
        image[image==0] = 255
        blur_image = cv2.GaussianBlur(image, (5, 5), 0)
        new_mask = np.zeros(image.shape, np.uint8)
        contours, hierachy = cv2.findContours(
            mask[:,:,0],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # for cnt in contours:
        #     print(cnt.shape)

        cv2.drawContours(new_mask, contours, -1, (255, 255, 255), 5)
        output = np.where(new_mask==np.array([255, 255, 255]), blur_image, image)
        # print(output)
        # output[output==0]=255
        cv2.imwrite(os.path.join(output_folder, mask_path.split('/')[-1]), output) 
        
        # cv2.imwrite("./out.png", mask*255) 
        # break