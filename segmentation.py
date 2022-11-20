
# Requirements: 
import argparse
import torch
import detectron2

# Import some common libraries
import cv2
import os
import glob
import numpy as np

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from pathlib import Path
import sys 

# Check versions. Expected: torch:  1.11 ; cuda:  1.11.0; detectron2: 0.6    
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Remove the extensions *.jpg when naming the *.csv file, needed to conserve the names for stitching back the crops
def get_filename_without_extension(data):
    file_basename = os.path.basename(data)
    filename_without_extension = file_basename.split('.')[0]
    return filename_without_extension

def run(
        weights=ROOT / "/content/weights_detectron2/model_final.pth",               # path to the weights
        output=ROOT / "/content/output",                                                     # folder where to store results
        data=ROOT / "/content/EDM-Craters/yolov5/runs/detect/exp/crops/Crater",     # folder to the images of single craters detected through YOLO v5
        conf_thres=0.80,                                                            # confidence threshold
        view_img=False,                                                             # show single crater image with mask
        save_img=False,                                                             # save single crater image with mask to *.jpg
        save_csv=False,                                                             # save masks to *.csv
        save_mask=False,                                                            # save masks to *.csv
):
    torch.multiprocessing.freeze_support()
    output = str(output)
    data= str(data)
    weights= str(weights)
    
    from detectron2.modeling import build_model
    cfg = get_cfg()
    model = build_model(cfg)

    # Inference with saved weigths
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # load the config file from the model zoo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1                                                                     # number of classes (here one: crater)                                               
    cfg.MODEL.WEIGHTS = (weights)                                                                           # path for the saved weigths
    cfg.DATASETS.TEST = ("my_dataset_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres                                                      # set the testing threshold for this model
    
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    acount=0
    
    for imageName in glob.glob(data + "/*.jpg"):  
        im = cv2.imread(imageName)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata, 
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW,
                        )
    
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Visualize the results: craters with mask
        if view_img:
            cv2.imshow('img',out.get_image()[:, :, ::-1])
            cv2.waitKey(0) #CHECK THISSSSS

        # Remove the extension .jpg to write on the csv file and save the mask
        get_filename_without_extension(imageName)
      
        # Convert the mask to a binary image (a matrix with False and True)
        p = outputs['instances'].pred_masks.cpu().numpy()

        # Convert the binary image to a matrix with 0 and 1 (Black and White)
        p = (np.where(p >= 1, 255, p))
        
        # Save the images with the mask
        if save_img:
            cv2.imwrite(os.path.join(output, os.path.basename(imageName)), out.get_image()[:, :, ::-1])

        # Save the matrix as a *.jpg image and as a *.csv file. The names are the same as the original images
        for yi in p:
            if save_mask: 
                cv2.imwrite(os.path.join(output , 'mask_' + os.path.basename(imageName)), yi)
            
            if save_csv:
                np.savetxt(os.path.join(output , os.path.basename(get_filename_without_extension(imageName)) + '.csv'), yi, delimiter=",")
        
        acount+=1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / "/content/weights_detectron2/model_final.pth", help='weights path')
    parser.add_argument('--output', type=str, default=ROOT / "/content/output", help='folder where to store results')
    parser.add_argument('--data', type=str, default=ROOT / "/content/EDM-Craters/yolov5/runs/detect/exp/crops/Crater", help='folder to the images of single craters detected through YOLO v5')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--view-img', action='store_true', help='show single crater image with mask')
    parser.add_argument('--save-img', action='store_true', help='save single crater image with mask to *.jpg')
    parser.add_argument('--save-csv', action='store_true', help='save masks to *.csv')
    parser.add_argument('--save-mask', action='store_true', help='save masks to *.jpg')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
