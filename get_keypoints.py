import numpy as np
import os
import cv2
import json
print('OpenCV - version: ',cv2.__version__)
import pandas as pd
import matplotlib.pyplot as plt

# Original input video file (no pose estimation on it)
cap = cv2.VideoCapture('D:\\Research footage\\Output\\boulder_1\\P2_trim.mp4')

#get the width and height of the input video
def get_vid_properties(cap): 
    width = int(cap.get(3))  
    height = int(cap.get(4)) 
    return width,height
  
print('Video Dimensions: ',get_vid_properties(cap))
width,height = get_vid_properties(cap)
cap.release()

# Load keypoint data from JSON output ('acc' = confidence score)
column_names = ['x', 'y', 'acc']

# Paths - should be the folder where Open Pose JSON output was stored
path_to_json = 'D:\\Research footage\\Ouput\\boulder_1\\P2_trim_json'

# Import Json files, pos_json = position JSON
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print('Found: ',len(json_files),'json keypoint frame files')
count = 0

# instantiate dataframes 
body_keypoints_df = pd.DataFrame()

print('json files: ',json_files[0])   

# Loop through all json files in output directory
# Each file is a frame in the video
# If multiple people are detected - choose the most centered high confidence points
for file in json_files:

    temp_df = json.load(open(path_to_json+"\\"+file))
    temp = []

    # (k,v) = (column label, content)
    #.items() iterates over columns
    for k,v in temp_df['part_candidates'][0].items():
        
        # Single point detected
        #each point = [x,y, confidence]
        if len(v) < 4:
            temp.append(v)
            #print('Extracted highest confidence points: ',v)
            
        # Multiple points detected
        elif len(v) > 4: 
            near_middle = width
            
            np_v = np.array(v)
            
            # Reshape to x,y,confidence
            #np_v_reshape = ([x,y,c],[x,y,c]....)
            np_v_reshape = np_v.reshape(int(len(np_v)/3),3)
            highest = np_v_reshape[0]
            np_v_temp = highest
            #compare y values
            
            for pt in np_v_reshape:
                if(pt[1]<highest[1]):
                    highest = np.array(pt)
                    np_v_temp = list(pt)
                
            temp.append(np_v_temp)
            
            """
            # compare x values
            
            for pt in np_v_reshape:
                if(np.absolute(pt[0]-width/2)<near_middle):
                    near_middle = np.absolute(pt[0]-width/2)
                    np_v_temp = list(pt)
            
            temp.append(np_v_temp)
            """
        else:
            # No detection - record zeros
            temp.append([0,0,0])
            
    temp_df = pd.DataFrame(temp)
    temp_df = temp_df.fillna(0)
    #print(temp_df)

    try:
        prev_temp_df = temp_df
        body_keypoints_df= body_keypoints_df.append(temp_df)
        #each joint is located in a specific row
    except:
        print('bad point set at: ', file)


#add column names        
body_keypoints_df.columns = column_names

#reset_index() labels the rows with 0,1,2... and not included as a column
body_keypoints_df.reset_index()

print('length of merged keypoint set: ',body_keypoints_df.size)

#remove the 'acc' column from the dataframe
body_keypoints_df.drop('acc', inplace=True, axis=1)

#save the keypoints to text file
body_keypoints_df.to_csv(r'D:\\Research footage\\Ouput\\boulder_1\\Keypoints\\P2_new_kpts.txt', index=False, header=False)
print("body_keypoints DataFrame: \n", body_keypoints_df)


