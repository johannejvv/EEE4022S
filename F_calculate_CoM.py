import numpy as np
import os
import cv2
import json
print('OpenCV - version: ',cv2.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
from matplotlib.ticker import MaxNLocator

X_path = 'D:\\Research footage\\Ouput\\boulder_2\\Keypoints\\P2_new_kpts.txt'
cap = cv2.VideoCapture('D:\\Research footage\\Input\\boulder_2\\P2_trim.mp4')
length = 1458
save_plot = "P2 plot.png"
plot_title = "Center of mass tracking "

#X_ is the array of all the keypoints
file = open(X_path, 'r')
X_ = np.array(
    [elem for elem in [
         row.split(',') for row in file
    ]], 
    dtype=np.float32
)
file.close()
#print(X_) 

#instantiate the empty joint centres
RWrist = np.empty((0,2))
RElbow =  np.empty((0,2))
RShoulder = np.empty((0,2)) 
LWrist = np.empty((0,2))
LElbow = np.empty((0,2))
LShoulder = np.empty((0,2)) 
RBigToe = np.empty((0,2))
RHeel = np.empty((0,2))
RAnkle = np.empty((0,2))
RKnee = np.empty((0,2))
RHip = np.empty((0,2))
LBigToe = np.empty((0,2))
LHeel = np.empty((0,2))
LAnkle = np.empty((0,2))
LKnee = np.empty((0,2))
LHip = np.empty((0,2))
Nose = np.empty((0,2))
Neck = np.empty((0,2))
MidHip = np.empty((0,2))
REar = np.empty((0,2))

##Get all the joint centre coordinates
i=0
while(i<length):
       
    RWrist = np.append(RWrist, np.array([X_[25*i + 4]]), axis=0)
    RElbow =  np.append(RElbow, np.array([X_[25*i +3]]), axis=0)
    RShoulder = np.append(RShoulder, np.array([X_[25*i +2]]), axis=0)
    LWrist = np.append(LWrist, np.array([X_[25*i +7]]), axis=0)
    LElbow = np.append(LElbow, np.array([X_[25*i +6]]), axis=0)
    LShoulder = np.append(LShoulder, np.array([X_[25*i +5]]), axis=0)
    RBigToe = np.append(RBigToe, np.array([X_[25*i +22]]), axis=0)
    RHeel = np.append(RHeel, np.array([X_[25*i +24]]), axis=0)
    RAnkle = np.append(RAnkle, np.array([X_[25*i +11]]), axis=0)
    RKnee = np.append(RKnee, np.array([X_[25*i +10]]), axis=0)
    RHip = np.append(RHip, np.array([X_[25*i +9]]), axis=0)
    LBigToe = np.append(LBigToe, np.array([X_[25*i +19]]), axis=0)
    LHeel = np.append(LHeel, np.array([X_[25*i +21]]), axis=0)
    LAnkle = np.append(LAnkle, np.array([X_[25*i +14]]), axis=0)
    LKnee = np.append(LKnee, np.array([X_[25*i +13]]), axis=0) 
    LHip = np.append(LHip, np.array([X_[25*i+12]]), axis=0)
    #Nose = np.append(Nose, np.array([X_[25*i+0]]), axis=0) 
    Neck = np.append(Neck, np.array([X_[25*i+1]]), axis=0)
    MidHip = np.append(MidHip, np.array([X_[25*i+8]]), axis=0)
    REar = np.append(REar, np.array([X_[25*i+17]]), axis=0)
    i=i+1

def find_CM(segment):
    #this finds the percentage at which the CM lies within each segment (just for females)
    switcher = {
         'Head': 0.4841,
         'Trunk': 0.3782,
         'UpperArm': 0.5754,
         'Forearm': 0.4559,
         'Thigh': 0.3612,
         'Shank': 0.4352,
         'Foot': 0.4014
        }
    return switcher.get(segment, "Invalid segment")

def segment_centr(a,b,segment):
    #this calculates the segment CM where a is the distal joint and b is the proximal joint
    #if one of two points is (0,0), centr = 0
    i=0
    centr=np.empty((0,2))
    while(i< a.shape[0]):
      
        if ((a[i].any() != 0) & (b[i].any() != 0) ):
            centre_temp = b[i] + find_CM(segment)*(a[i]-b[i])  
            centr=np.append(centr, np.array([centre_temp]), axis=0)
            i=i+1
       
        elif (i>0):
            centr = np.append(centr, np.array([centr[i-1]]), axis=0)
            i=i+1
        else:
            centr = np.append(centr, np.array([[0,0]]), axis=0)
            i=i+1
          
    return (centr)

##Get all the segment centres
RForearm_centr = segment_centr(RWrist, RElbow, 'Forearm')
RUpperarm_centr = segment_centr(RElbow, RShoulder, 'UpperArm')
LForearm_centr = segment_centr(LWrist, LElbow, 'Forearm')
LUpperarm_centr = segment_centr(LElbow, LShoulder, 'UpperArm')
RFoot_centr = segment_centr(RBigToe, RHeel, 'Foot')
RShank_centr = segment_centr(RAnkle, RKnee, 'Shank')
RThigh_centr = segment_centr(RKnee, RHip, 'Thigh')
LFoot_centr = segment_centr(LBigToe, LHeel, 'Foot')
LShank_centr = segment_centr(LAnkle, LKnee, 'Shank')
LThigh_centr = segment_centr(LKnee, LHip, 'Thigh')
Head_centr = segment_centr(REar, Neck, 'Head')
Trunk_centr = segment_centr(Neck, MidHip, 'Trunk')

def find_mass(segment):
    #this finds the proprotion of mass that each segment makes of the total mass (just for females)
    switcher = {
         'Head': 0.0668,
         'Trunk': 0.4258,
         'UpperArm': 0.0255,
         'Forearm': 0.0138,
         'Thigh': 0.1478,
         'Shank': 0.0481,
         'Foot': 0.0129
        }
    return switcher.get(segment, "Invalid segment")

def seg_moment(segment_centr, segment):
    #this calculates the segment moment about the segment center
    moment = segment_centr*find_mass(segment)
    return moment

def body_CM():

    RForearm_mom = seg_moment(RForearm_centr, 'Forearm')
    RUpperarm_mom = seg_moment(RUpperarm_centr, 'UpperArm')
    LForearm_mom = seg_moment(LForearm_centr, 'Forearm')
    LUpperarm_mom = seg_moment(LUpperarm_centr, 'UpperArm')
    RFoot_mom = seg_moment(RFoot_centr, 'Foot')
    RShank_mom = seg_moment(RShank_centr, 'Shank')
    RThigh_mom = seg_moment(RThigh_centr, 'Thigh')
    LFoot_mom = seg_moment(LFoot_centr, 'Foot')
    LShank_mom = seg_moment(LShank_centr, 'Shank')
    LThigh_mom = seg_moment(LThigh_centr, 'Thigh')
    Trunk_mom = seg_moment(Trunk_centr, 'Trunk')
    Head_mom = seg_moment(Head_centr, 'Head')

    COM = RForearm_mom + RUpperarm_mom + LForearm_mom + LUpperarm_mom + RFoot_mom + RShank_mom + RThigh_mom + LFoot_mom + LShank_mom + LThigh_mom +Trunk_mom + Head_mom

    return COM

COM = body_CM()

#smooth out the COM_y
i=1
COM_smooth=np.empty((0,2))
COM_smooth = np.append(COM_smooth, np.array([COM[0]]),axis=0)

while(i< COM.shape[0]):
    check = np.linalg.norm(COM[i]-COM_smooth[i-1])
    #print(check)
    if(check>2000):
        COM_smooth = np.append(COM_smooth, np.array([COM_smooth[i-1]]),axis=0)
        i=i+1
    else:
        COM_smooth = np.append(COM_smooth, np.array([COM[i]]),axis=0)
        i=i+1
#print(COM_smooth)
savetxt('COM.csv', COM_smooth, delimiter=',')

#setup OpenCv to create video output
out = cv2.VideoWriter(
    'output.mp4',
   cv2.VideoWriter_fourcc(*'mp4v'),
    30.0,
    (540,960))


#create video output of the CoM
i=0
while True:
    success, frame = cap.read()

    if success == True:
        scale_factor = 4

        width = int(frame.shape[1]/scale_factor)
        height = int(frame.shape[0]/scale_factor)
        dim = (width, height)
        frame= cv2.resize(frame, dim)

        if (i<length):
            pt = (int((COM_smooth[i][0])/scale_factor), int((COM_smooth[i][1])/scale_factor))
            frm = cv2.circle(frame, pt, 10, (255, 0, 0), -1)
            i=i+1
        else:
            i=0
            frm = cv2.circle(frame, pt, 10, (255, 0, 0), -1)
       
        # Write the output video 
        out.write(frm.astype('uint8'))
        cv2.imshow("Video", frm)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    else:
        break
    
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)

#plot the COM 
i=0
fig,ax = plt.subplots()

for i in range(length):
    ax.plot(int(COM_smooth[i][0]),int(COM_smooth[i][1]),color='blue', marker='.') 
        
plt.xlabel("x position [pixels]")
plt.ylabel("y position [pixels]")
plt.title(plot_title)
plt.gca().invert_yaxis()
plt.savefig(save_plot)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(length):
    ax1.plot(int(COM[i][0]),int(COM[i][1]), color='blue', marker='.') 

for i in range(length):
    ax2.plot(int(COM_smooth[i][0]),int(COM_smooth[i][1]), color='blue', marker='.') 

#plt.gca().invert_yaxis()
plt.ylim(60,3500)
plt.xlim(-60,2125)
ax1.invert_yaxis()
ax2.invert_yaxis()

ax1.set_title('COM tracking before smoothing')
ax2.set_title('COM tracking after smoothing')

#x, y = zip(*COM_smooth)
#plt.scatter(x, y, None, None, ".")
plt.show() 
