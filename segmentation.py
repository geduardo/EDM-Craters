
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
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
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
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

 # To remove the extensions .jpg when writing on the csv file
def get_filename_without_extension(data):
    file_basename = os.path.basename(data)
    filename_without_extension = file_basename.split('.')[0]
    return filename_without_extension

def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        output=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        conf_thres=0.25,  # confidence threshold
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_img=False,
        save_mask=False,
        nosave=False,  # do not save images/videos
):
    torch.multiprocessing.freeze_support()
    output = str(output)
    data= str(data)

    print(weights)
    weights= str(weights)
    print(weights)
    # Directories
    save_img = not nosave
    save_txt = not nosave
    save_mask = not nosave
    from detectron2.modeling import build_model
    cfg = get_cfg()
    model = build_model(cfg)
    print(output)

    # Inference with saved weigths
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = (weights) # path for the saved weigths
    cfg.DATASETS.TEST = ("my_dataset_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres # set the testing threshold for this model
    
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    acount=0

    for imageName in glob.glob(os.path.join(data,'*jpg')):  
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
            cv2.waitKey(0)

        # Remove the extension .jpg to write on the csv file and save the mask
        get_filename_without_extension(imageName)
      
         # Convert the mask to a binary image (a matrix with False and True)
        p=outputs['instances'].pred_masks.cpu().numpy()
        #print(p) 

        # Convert the binary image to a matrix with 0 and 1 (Black and White)
        p=(np.where(p>=1, 255, p))
        
        # Save the images with the mask
        if save_img:
            cv2.imwrite(os.path.join(output, os.path.basename(imageName)), out.get_image()[:, :, ::-1])

        # Save the matrix as a .jpg image and as a .csv file. The names are the same as the original images
        for yi in p:
            #cv2.imwrite('mask_'+get_filename_without_extension(imageName)+'.jpg', yi)
            if save_mask: 
                cv2.imwrite(os.path.join(output , 'mask_' + os.path.basename(imageName) +'.jpg'), yi)
            
            #np.savetxt(get_filename_without_extension(imageName)+'.csv', yi, delimiter=",")
            if save_txt:
                np.savetxt(os.path.join(output , os.path.basename(get_filename_without_extension(imageName)) +'.csv'), yi, delimiter=",")
        
        # Compute area of the craters by counting the number of white pixels and check if the total number of pixel counted is same as the size of the image as it should be  
        # print(p.size)
        # number_of_white_pix = np.sum(p == 255)
        # number_of_black_pix = np.sum(p == 0)
        # print('NUMBER OF WHITE PIXELS:',  number_of_white_pix)
        # print('NUMBER OF BLACK PIXELS:',  number_of_black_pix)
        # total=number_of_black_pix+number_of_white_pix
        # print(total)
        acount+=1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--output', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-img', action='store_true', help='save results to *.jpg')
    parser.add_argument('--save-mask', action='store_true', help='save results to *.jpg')
    parser.add_argument('--nosave', action='store_true', help='do not save images')
    #parser.add_argument('--area', action='store_true', help='do not save images')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)