import cv2
import os

def get_image_from_video(video_path):
    # Return path to 10th frame in video
    cam = cv2.VideoCapture(video_path) 
    try: 
        if not os.path.exists('tmp_image_data'): 
            os.makedirs('tmp_image_data')
    except OSError: 
        print ('Error: Creating directory of tmp image data') 
    
    currentframe = 0
    frame_to_get = 10

    while(currentframe <= frame_to_get): 
        ret,frame = cam.read() 
    
        if (currentframe == frame_to_get) and ret: 
            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
    
            # writing the extracted images 
            cv2.imwrite(name, frame) 
            
            return name
        else: 
            break
    return None

def target_instance(target_vid_path):
    return get_image_from_video(target_vid_path)

def base_instance(base_vid_path):
    return get_image_from_video(base_vid_path)