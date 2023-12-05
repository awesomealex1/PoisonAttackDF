import cv2
import os
import time

def get_image_from_video(video_path):
    video_path = 'test.mp4'
    # Return path to 10th frame in video
    cam = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    print(cam.isOpened())
    print(video_path)
    print(cam.getBackendName())
    try: 
        if not os.path.exists('tmp_image_data'): 
            os.makedirs('tmp_image_data')
    except OSError: 
        print ('Error: Creating directory of tmp image data') 
    
    currentframe = 0
    frame_to_get = 1000

    success,image = cam.read()
    print("AA", success)
    count = 0
    while success:
        success,image = cam.read()
        print('Read a new frame: ', success)
        count += 1

    while(False): 
        ret,frame = cam.read() 
        print(ret)
#        if (currentframe == frame_to_get) and ret: 
            # if video is still left continue creating images 
#            name = './data/frame' + str(currentframe) + '.jpg'
#            print ('Creating...' + name) 
    
            # writing the extracted images 
#            cv2.imwrite(name, frame) 
            
#            return frame
        currentframe += 1
    return image

def target_instance(target_vid_path):
    return get_image_from_video(target_vid_path)

def base_instance(base_vid_path):
    return get_image_from_video(base_vid_path)

def img_path_to_tensor(img_path):
    return None

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb