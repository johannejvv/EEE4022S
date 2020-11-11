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


X_path = 'D:\\Research footage\\Ouput\\boulder_2\\Keypoints\\P3_new_kpts.txt'
cap = cv2.VideoCapture('D:\\Research footage\\Input\\boulder_2\\P3_trim.mp4')
length = 717
save_plot = "P2 plot.png"
plot_title = "Center of mass tracking"

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

i=0

while(i<length):
    #sum_temp = all_keypoints_arr[25*i:25*i + 24].sum(axis=0)
    #print("centr_temp:",centr_temp)
    #sum_all = np.append(sum_all, np.array([sum_temp]), axis=0)
    
    ##Get all the joint centre coordinates
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
        """
        elif (a[i].any() == 0):
            centr=np.append(centr, np.array([b[i]]), axis=0)
            i=i+1
        elif (b[i].any() == 0):
            centr=np.append(centr, np.array([a[i]]), axis=0)
            i=i+1
        """
        
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

Trunk_centr = segment_centr(Neck, MidHip, 'Trunk')
Head_centr = segment_centr(REar, Neck, 'Head')


def find_mass(segment):
    #this finds the proprotion of mass that each segment makes of the total mass
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
#print("COM:", COM)
# save to csv file
#savetxt('P2_COM.csv', COM, delimiter=',')

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




"""
i=0
COM_y = []
COM_x =[]
while (i<length):
    #np.append(RWrist, np.array([X_[25*i + 4]]), axis=0)
    COM_y = np.append( COM_y, COM_smooth[i][1])
    COM_x = np.append( COM_x, COM_smooth[i][0])
    i=i+1

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.03

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 9))

ax_scatter = plt.axes(rect_scatter)
plt.gca().invert_yaxis()
plt.xlabel("x position [pixels]")
plt.ylabel("y position [pixels]")

ax_scatter.tick_params(direction='in', top=True, right=True)
plt.xlim(850,1600)
plt.ylim(3550,200)
#ax_scatter.xaxis.set_major_locator(MaxNLocator(prune='both'))
#ax_scatter.yaxis.set_major_locator(MaxNLocator(prune='both'))


ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histx.set_yticklabels([])
ax_histx.set_xticklabels([])
#ax_histx.yaxis.set_major_locator(MaxNLocator(4, prune='both'))
sns.distplot(COM_x, rug=False, hist=False, color='gray')

ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)
plt.gca().invert_yaxis()
ax_histy.set_ylim(ax_scatter.get_ylim())
ax_histy.set_yticklabels([])
ax_histy.set_xticklabels([])
#ax_histy.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
sns.distplot(COM_y, rug=False, hist=False, vertical=True, color='gray')

# the scatter plot:
for i in range(length):
    ax_scatter.plot(int(COM_smooth[i][0]),int(COM_smooth[i][1]),color='gray', marker='.') 
#ax_scatter.plot(COM_x, COM_y, color='orange', marker='.')


#ax_histx.set_xlim(ax_scatter.get_xlim())
#ax_histy.set_ylim(ax_scatter.get_ylim())
plt.show()
"""
"""
#print(COM_y)
#Display COG on video
subplot(2,2,4)
ax = sns.distplot(COM_y, rug=False, hist=False, vertical=True )
plt.gca().invert_yaxis()

subplot(2,2,1)
ax = sns.distplot(COM_x, rug=False, hist=False)

subplot(2,2,3)
ax = plt.plot(COM_x,COM_y, ".") 
plt.gca().invert_yaxis()


plt.show()
"""

out = cv2.VideoWriter(
    'output.mp4',
   cv2.VideoWriter_fourcc(*'mp4v'),
    30.0,
    (540,960))
"""
i=0
while True:
    success, frame = cap.read()

    if success == True:
        scale_factor = 4

        width = int(frame.shape[1]/scale_factor)
        height = int(frame.shape[0]/scale_factor)
        dim = (width, height)
        frame= cv2.resize(frame, dim)
    
   
        #pt = (200,600)
        
    
        if (i<length):
            pt = (int((COM_smooth[i][0])/scale_factor), int((COM_smooth[i][1])/scale_factor))
            frm = cv2.circle(frame, pt, 10, (255, 0, 0), -1)
            i=i+1
        else:
            i=0
            frm = cv2.circle(frame, pt, 10, (255, 0, 0), -1)
        #cv2.imshow("mid_hip",frame)
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
"""


i=0
#plot the COM 

fig,ax = plt.subplots()

for i in range(length):
    ax.plot(int(COM_smooth[i][0]),int(COM_smooth[i][1]),color='blue', marker='.') 
        
plt.xlabel("x position [pixels]")
plt.ylabel("y position [pixels]")
plt.title(plot_title)
plt.gca().invert_yaxis()
plt.savefig(save_plot)
plt.show()

"""
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
"""