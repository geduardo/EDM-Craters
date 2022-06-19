
# Requirements: 

import torch
import detectron2

# Import some common libraries
import cv2
import json
import os
import glob
import random
import numpy as np

# Check versions. Expected: torch:  1.11 ; cuda:  1.11.0; detectron2: 0.6

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
    
import os

# Multiprocessing

def run():
    torch.multiprocessing.freeze_support()
    print('loop')


Path_to_data=input("Path to data? ") # /Users/davidesa/Downloads/Craters EDM.v4i.coco_2/test/
Path_to_result_folder=input("Path to result folder? ") # C:/Users/davidesa/Documents

# Directory
directory = "Results"
  
# Parent Directory path
parent_dir = "C:/Users/davidesa/Documents"
  
# Path
path = os.path.join(Path_to_result_folder, directory)
  
# Create the directory
# 'results' in
# '/home / User / Documents'
os.mkdir(path)
print("Directory '% s' created" % directory)


if __name__ == '__main__':
    run()

    from detectron2.modeling import build_model
    cfg = get_cfg()
    model = build_model(cfg)

    # Inference with saved weigths
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = ("/Users/davidesa/output/model_final.pth") # path for the saved weigths
    cfg.DATASETS.TEST = ("my_dataset_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90 # set the testing threshold for this model
    
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    print("DONE INFERENCE")
    # To remove the extensions .jpg when writing on the csv file
    def get_filename_without_extension(file_path):
        file_basename = os.path.basename(file_path)
        filename_without_extension = file_basename.split('.')[0]
        return filename_without_extension
    print("DONE INFERENCE")
    
    
    # roboflowchar=43
    # # Path to test files
    # path="/Users/davidesa/WEDMSC_Mask-2/test/"
    # print("DONE")
    # # Exclude last eleemnt of the list (_annotation.coco) and rename files
    # for i in os.listdir(path)[:-1]:
    #     length_name=len(i)
    #     newlength=length_name-roboflowchar
    #     os.rename(os.path.join(path, i),os.path.join(path, i[:newlength])+'.jpg')

    from detectron2.utils.visualizer import ColorMode
    import glob
    print("DONE INFERENCE")
    acount=0

    # Insert path for test files '/Users/davidesa/Downloads/Craters EDM.v4i.coco_2/test/*jpg'
    for imageName in glob.glob(os.path.join(Path_to_data,'*jpg')):   # /Users/davidesa/WEDMSC_Mask-2/test/*jpg
        im = cv2.imread(imageName)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata, 
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW,
                        )
    
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Visualize the results: craters with mask
        cv2.imshow('img',out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

        # Print name of the crater and write on the images
        print(get_filename_without_extension(imageName))
        #cv2.imwrite(str(acount)+'result.jpg', out.get_image()[:, :, ::-1])
        #cv2.imwrite(os.path.basename(imageName), out.get_image()[:, :, ::-1])
        cv2.imwrite(os.path.join(path, os.path.basename(imageName)), out.get_image()[:, :, ::-1])
    
        #print(outputs["instances"].pred_masks)

        # Convert the mask to a binary image (a matrix with False and True)
        p=outputs['instances'].pred_masks.cpu().numpy()
        #print(p) 

        # Convert the binary image to a matrix with 0 and 1 (Black and White)
        p=(np.where(p>=1, 255, p))
        
        # Save the matrix as a .jpg image and as a .csv file. The names are the same as the original images
        for yi in p:
            #cv2.imwrite('mask_'+get_filename_without_extension(imageName)+'.jpg', yi)
            
            cv2.imwrite(os.path.join(path , 'mask_' + os.path.basename(imageName) +'.jpg'), yi)
            
            #np.savetxt(get_filename_without_extension(imageName)+'.csv', yi, delimiter=",")
            
            np.savetxt(os.path.join(path , os.path.basename(imageName) +'.csv'), yi, delimiter=",")
        
        # Compute area of the craters by counting the number of white pixels and check if the total number of pixel counted is same as the size of the image as it should be  
        print(p.size)
        number_of_white_pix = np.sum(p == 255)
        number_of_black_pix = np.sum(p == 0)
        print('NUMBER OF WHITE PIXELS:',  number_of_white_pix)
        print('NUMBER OF BLACK PIXELS:',  number_of_black_pix)
        total=number_of_black_pix+number_of_white_pix
        print(total)
        acount+=1
