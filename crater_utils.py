# import all the libraries
import os
import cv2
import shutil
import pandas as pd
import numpy as np

def crop(px_size =1.7, offset_x=0, offset_y=0):
    # create folder called crops
    if offset_x == 0 and offset_y == 0:
        output_directory = "crops"
    if offset_x != 0 and offset_y == 0:
        output_directory = "crops_x"
    if offset_x == 0 and offset_y != 0:
        output_directory = "crops_y"
    if offset_x != 0 and offset_y != 0:
        output_directory = "crops_xy"
        
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    directory  = './target_image'
    for filename in os.scandir(directory):
        if filename.is_file():
            if filename.name.endswith('.jpg') or filename.name.endswith('.png'):
                image = cv2.imread(filename.path)
                h, w = image.shape[:2]
                # crop with offset
                image = image[offset_y:h, offset_x:w]
                print("Height = {}, Width = {}".format(h, w))
                N = int(h/1000)
                M = int(w/1000)
                pixel_size = float(px_size)
                # add black pixels to the image to make it divisible by 1000
                image = cv2.copyMakeBorder(image, 0, 1000 - h%1000, 0, 1000 - w%1000, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                for i in range(N):
                    for j in range(M):
                        crop_img = image[i*1000:i*1000+1000, j*1000:j*1000+1000]
                        # add a scale bar of 100 micrometers
                        L = int(100/pixel_size)
                        x_pos = 900 - L
                        cv2.rectangle(crop_img, (900, 950), (x_pos, 955), (0, 40, 255), -1)
                        # Add text on top of the scale bar
                        text_pos = (int(x_pos), 940)
                        cv2.putText(crop_img, "100um", text_pos, cv2.QT_FONT_NORMAL, 0.75, (255, 255, 255), 2) 
                        cv2.imwrite(output_directory + '/'+ filename.name[:-4] + '_' + str(i) + '_' + str(j)  + '_' '.jpg', crop_img)
    
def separate_labels(path, output_path, delta_x=0, delta_y=0):
    os.makedirs(output_path, exist_ok=True)
    for file in os.listdir(path):
        if file.endswith(".txt"):
            # Open the file
            f = open(path + file, "r")
            # for each line in the file, extract the label and save it in a new file
            i = 0
            for line in f:
                # Extract the label and save it in a new file and substitute the x
                # and y relative coordinates of the crop with the absolute
                # coordinates of the original image
                #get coordinates from the filename
                # filename format: image_name_i_j_k.txt where i and j are the row
                # and column of the crop
                container = file.split('_')
                x_offset = int(container[2])*1000 + delta_x
                y_offset = int(container[1])*1000 + delta_y
                label = line.split('\n')[0]
                label = label.split(' ')
                label[1] = str(int(1000*float(label[1])) + x_offset)
                label[2] = str(int(1000*float(label[2])) + y_offset)
                label[3] = str(int(1000*float(label[3])))
                label[4] = str(int(1000*float(label[4])))
                label = ' '.join(label)
                # only save labels if ratio of the bounding box is not too small or too big
                if int(label.split(' ')[4])/int(label.split(' ')[3]) > 0.75 and int(label.split(' ')[4])/int(label.split(' ')[3]) < 1.5:
                    with open(output_path + file.split('.')[0] + str(i) + '.txt', 'w') as f1:
                        f1.write(label)
                        f1.close()
                i += 1
                    
def plot_bounding_boxes(df, output_path, image_path= './target_image/', color = (0, 0, 255), thickness = 3):
    for file in os.listdir(image_path):
        img = cv2.imread(image_path + file)
        name = file.split('.')[0]
    # store the labels in a list
    labels = []
    columns = ['x', 'y', 'w', 'h']
    for i in range(0, len(df)):
        labels.append(list(df.loc[i, columns]))
    for label in labels:
        x = int(label[0])
        y = int(label[1])
        w = int(label[2])
        h = int(label[3])
        cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, thickness)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(output_path + name + '_labeled.jpg', img)
    
    
def crop_craters_from_df(df, output_path, image_path= './target_image/',):
    for file in os.listdir(image_path):
        img = cv2.imread(image_path + file)
        name = file.split('.')[0]
    # store the labels in a list
    labels = []
    columns = ['id', 'x', 'y', 'w', 'h']
    # create a column with the id of the crater
    df['id'] = df.index
    for i in range(0, len(df)):
        labels.append(list(df.loc[i, columns]))
    os.makedirs(output_path, exist_ok=True)
    
    for label in labels:
        id = str(label[0])
        x = int(label[1])
        y = int(label[2])
        w = int(label[3])
        h = int(label[4])
        # crop the crater and save it
        crop_img = img[y-h//2:y+h//2, x-w//2:x+w//2]
        cv2.imwrite(output_path + name + '_' + id + '.jpg', crop_img)

def create_dataframe():
    
    x = ['0', 'x', 'y', 'xy']
    labels = []
    for i in x:
        path = './results_' + i + '/separated_labels/'
        for file in os.listdir(path):
            if file.endswith(".txt"):
                f = open(path + file, "r")
                for line in f:
                    line = line.split('\n')[0]
                    line = line.split(' ')
                    line.pop(0)
                    for j in range(0,4):
                        line[j] = round(int(line[j]))
                    labels.append(line)


    # remove duplicates. We define as a duplicate a label that has coordinates that
    # are at an euclidean distance of less than 20 pixels from any other label. 
    # We decide to keep the label with the biggest area

    aux_list = []
    for label in labels:
        duplicates = []
        x, y, w, h = int(label[0]), int(label[1]), int(label[2]), int(label[3])
        for label2 in labels:
            x2, y2, w2, h2 = int(label2[0]), int(label2[1]), int(label2[2]), int(label2[3])
            distance = np.sqrt((x-x2)**2 + (y-y2)**2)
            if distance < 20:
                duplicates.append(label2)
                
        aux_list.append(duplicates)

    # from the aux_list we select the label with the biggest area
    good_labels = []
    for element in aux_list:
        if len(element) == 1:
            good_labels.append(element[0])
        else:
            areas = []
            for label in element:
                areas.append(int(label[2])*int(label[3]))
            good_labels.append(element[np.argmax(areas)])
    
    # delete duplicates from the good_labels list
    good_labels = [list(x) for x in set(tuple(x) for x in good_labels)]
    
    # create a dataframe
    
    df = pd.DataFrame()    
    df = df.append(good_labels)
    df.columns = ['x', 'y', 'w', 'h']
    return df