# EEE4022S

This repository contains the Python and MATLAB code prepared for the 4022S final year project by Johanné Jansen van Vuuren (JNSJOH021) at the University of Cape Town.

The scripts included are as follows:
* **get_keypoints.py** takes the OpenPose generated JSON file to create a single text file containing all the skeleton keypoints.
* **calculate_CoM.py** uses the skeleton keypoints to calculate the center of mass (CoM) and will export the CoM values in a .csv file.
  * There are two versions of this program: **M_calculate_CoM.py** is to calculate the CoM for males and **F_calculate_CoM.py** is for females. 
* **get_averages.m** calculates the average velocity, acceleration and power of the CoM trajectory
* **vel_acc_covariance.m** calculates and plots the covariance values for velocity and acceleration
* **pwr_covariance.m** calculates and plots the covariance values for power

### Prerequisites:
The Python programs require the OpenCV library to be installed.
