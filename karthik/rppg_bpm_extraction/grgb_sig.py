'''Program to extract specific ROIs from the video frame. 
Compute the average values of color channels for the selected ROIs.

Retruns the GRGB signal and fps of the video

Mediapipe library is used to detect the face landmarks'''

import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest

def grgb_est(video_path):

    # # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Drawing utility
    #mp_drawing = mp.solutions.drawing_utils

    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    #ROI loction points according to the paper by F.Haugg et al. (added additional points in each case)
    landmark_head = [107, 66, 69, 67, 109, 10, 338, 297, 299, 296, 336, 9]
    landmark_lcheek = [118, 119, 120, 47, 126, 209, 49, 129, 203, 205, 50]
    landmark_rcheek = [347, 348, 349, 277, 355, 429, 279, 358, 423, 425, 280]

    #Initialise the list of color channels
    avg_b = []
    avg_g = []
    avg_r = []

    """Set the time durtation of video you want to """
    #To set the total time duration of video interested in
    time_period = 20   #seconds
    #Set the value (= total frame - to perform for full length or 0 for any specific time duration)
    f_count =  total_frames
    #f_count =  0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)
        #print(len(results.multi_face_landmarks))

        # Extract the values of specific ROI with respect to landmark position points
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #print(len(face_landmarks.landmark))
                
                # # Extracting landmark coordinates
                # h, w, _ = frame.shape
                # landmark_coords = []
                # for landmark in face_landmarks.landmark:
                #     x = int(landmark.x * w)
                #     y = int(landmark.y * h)
                #     z = landmark.z
                #     landmark_coords.append((x, y, z))
                
                # Extracting specific landmark coordinates
                h, w, _ = frame.shape
                landmark_coords_head = []
                landmark_coords_lcheek = []
                landmark_coords_rcheek = []

                for i, j, k in zip_longest(landmark_head, landmark_lcheek, landmark_rcheek, fillvalue = None):

                    if i is not None:
                        landmark_h = face_landmarks.landmark[i]
                        x = int(landmark_h.x * w)
                        y = int(landmark_h.y * h)
                        landmark_coords_head.append((x, y))
                        #Mark the landamrk points on the image frame
                        cv2.circle(frame, (x, y), radius=1, color=(0, 0, 0), thickness=1)
                    
                    if j is not None:

                        landmark_lc = face_landmarks.landmark[j]
                        x1 = int(landmark_lc.x * w)
                        y1 = int(landmark_lc.y * h)
                        landmark_coords_lcheek.append((x1, y1))
                        #Mark the landamrk points on the image frame
                        cv2.circle(frame, (x1, y1), radius=1, color=(0, 0, 0), thickness=1)
                    
                    if k is not None:

                        landmark_rc = face_landmarks.landmark[k]
                        x2 = int(landmark_rc.x * w)
                        y2 = int(landmark_rc.y * h)
                        landmark_coords_rcheek.append((x2, y2))
                        #Mark the landamrk points on the image frame
                        cv2.circle(frame, (x2, y2), radius=1, color=(0, 0, 0), thickness=1)
                    

                # Determine the bounding box around the specified landmarks -
                # Forehead
                x_coords, y_coords = zip(*landmark_coords_head)
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                # ROI
                forehead_roi = frame[y_min:y_max, x_min:x_max]
                forehead_blue = np.mean(forehead_roi[:, :, 0])
                forehead_green = np.mean(forehead_roi[:, :, 1])
                forehead_red = np.mean(forehead_roi[:, :, 2])

                #lcheek
                x_coords, y_coords = zip(*landmark_coords_lcheek)
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                # ROI
                lcheek_roi = frame[y_min:y_max, x_min:x_max]
                lcheek_blue = np.mean(lcheek_roi[:, :, 0])
                lcheek_green = np.mean(lcheek_roi[:, :, 1])
                lcheek_red = np.mean(lcheek_roi[:, :, 2])

                #rcheek
                x_coords, y_coords = zip(*landmark_coords_rcheek)
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                # ROI
                rcheek_roi = frame[y_min:y_max, x_min:x_max]
                rcheek_blue = np.mean(rcheek_roi[:, :, 0])
                rcheek_green = np.mean(rcheek_roi[:, :, 1])
                rcheek_red = np.mean(rcheek_roi[:, :, 2])

                # Crop the image
                #cropped_image = frame[y_min:y_max, x_min:x_max]
            
                #average = np.mean(forehead_roi[:, :, 0], lcheek_roi[:, :, 0], rcheek_roi[:, :, 0])
                #print(forehead_red, lcheek_red, rcheek_red)
                
                avg_r.append(np.mean([forehead_red, lcheek_red, rcheek_red]))
                avg_g.append(np.mean([forehead_green, lcheek_green, rcheek_green]))
                avg_b.append(np.mean([forehead_blue, lcheek_blue, rcheek_blue]))

                #r, g, b = frame[y_min:y_max, x_min:x_max]
                #print(f"R G B values are : {avg_r}, {avg_g}, {avg_b}")
                
                # # Display the cropped image in a separate window
                # cv2.namedWindow("Cropped_Image", cv2.WINDOW_NORMAL) 
                # cv2.resizeWindow("Cropped_Image", 400,200)
                # cv2.imshow('Cropped_Image', cropped_image)
                
        # Display the frame
        #cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL) 
        #cv2.resizeWindow("Face Landmarks", 900,650)
        #cv2.imshow('Face Landmarks', frame)
        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # To retrieve no. of frames according to the specified time duration
        if f_count == total_frames:
            continue
        else:
        
            if(f_count/fps >= time_period):
                print("Current frame counts :",f_count)
                break
            else:
                f_count+=1
            
    #print("Total No. of vid frames: ", total_frames)
    #print("Video FPS: ", fps)

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    # Calculate the GR-GB values of the frames
    #Implementation of the method mentioned in paper by F.Haugg et al.
    gr_signal = [a/b for a, b in zip(avg_g, avg_r)]

    gb_signal = [a/b for a, b in zip(avg_g, avg_b)]

    grgb_signal = [a + b for a , b in zip(gr_signal, gb_signal)]
    
    return grgb_signal, fps
