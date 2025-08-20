import cv2, os, random
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tracker import bb_tracker
from video import video_editor
from time import time
import supervision as sv
import argparse
import yaml

def id_to_color(id):
    """
    Function to convert an id to a unique color

    Parameters
    ----------
    id : int
        The ID
    """
    blue  = id*107 % 256
    green = id*149 % 256
    red   = id*227 % 256
    return (red, green, blue)

def drawPred(frame, type, id, conf, box, color=(255, 0, 0)):
    '''
    Draws a bounding box around the detected object 

    Parameters
    ----------
    frame : array[width, height, color depth]
        The name of the video
    type : str
        The object type
    conf : float
        A value between 0 and 1 indicating the confidence of the detection
    box : array[4]
        The bounding box
    color : bool
        The color of the bounding box
    '''

    left   = int(box[0])
    top    = int(box[1])
    right  = int(box[2])
    bottom = int(box[3])

    # Draw the bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=5)

    # Create label with type and confidence
    label = "{}({}): {:.2f}".format(type, id, conf)

    # Display the label at the top of the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 3
    color = (255,255,255)
    labelSize, _ = cv2.getTextSize(label, font, scale, thickness)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), font, scale, color, thickness, cv2.LINE_AA)

    return frame

def get_config(config_file=None):
    """ Merges YAML config with command line arguments

    Returns:
        object: the configuration
    """

    if config_file is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Parser for SORT Tracking')
        parser.add_argument('-c', '--config',   default='config.yaml', help='Path to YAML config file')

        args = parser.parse_args()

        config_file = args.config

        # Open config file
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Update config with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        # Open config file
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    return config

def main(config, create_video=True, realtime_display=False):

    if config['video'] == 'RANDOM':
        # Get a random test image (dataset used is BDD100k)
        video_file = random.choice(os.listdir(config['video_dir']))
    else:
        video_file = config['video'] #'cd20d7e6-dc9d2d27.mov'

    video_path = f'videos/test/{video_file}'

    # Initialize YOLO model
    YOLO_MODEL = 'yolo12n'
    model = YOLO('{}.pt'.format(YOLO_MODEL))

    # Create dictionary for trackers
    trackers = {}

    # Load the video and process it one frame at a time
    vidcap = cv2.VideoCapture(video_path)
    images_to_process, img = vidcap.read()

    if create_video:
        video = video_editor('out', video_file, width=int(img.shape[1]/4), height=int(img.shape[0]/4))

    if realtime_display:
        # Display the first image
        fig, ax = plt.subplots()
        imgplot = ax.imshow(img)
        plt.show(block=False)

    num_frames = 0
    start = time()
    
    while images_to_process:

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_frames+=1

        # Detect objects with YOLO
        yolo_result = model.predict(img, verbose=False)

        detections = sv.Detections.from_ultralytics(yolo_result[0])
        detected_classes = set(detections.class_id)
        for cls in detected_classes:
            if cls not in trackers:
                trackers[cls] = bb_tracker(config)

            # Process the detections and get all matches
            trackers[cls].process_detections(detections[detections.class_id == cls], img.shape)
            (ids, boxes, confidences) = trackers[cls].get_matches()

            if create_video:
                # Draw matches on the frame and add it to the video
                for id, box, confidence in zip(ids, boxes, confidences):
                    type = yolo_result[0].names[cls]
                    img = drawPred(img, type, id, confidence, box, id_to_color(id))

            if realtime_display:
                # Update the image with the freshly annotated image
                imgplot.set_data(img)
                plt.draw()
                plt.pause(0.001)

        video.add_frame(img)
        images_to_process, img = vidcap.read()

        # For Debug
        #tracker.print_matches()
        #tracker.print_unmatched_trackers()
        #tracker.print_unmatched_detections()

    if create_video:
        # Save the video
        video.save_video()

    # Calculate FPS
    average_fps = num_frames / (time()-start)
    print(f'FPS: {average_fps:.2f}')

if __name__ == '__main__':
    config = get_config()
    main(config['TRACKER'], create_video=config['DEBUG']['create_video'], realtime_display=config['DEBUG']['realtime_display'])
