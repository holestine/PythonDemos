import cv2, os, random
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tracker import bb_tracker
from video import video_editor
from time import time

def main(video_path=None, create_video=True, realtime_display=False):

    # Validate input
    if video_path == None:
        print('Please specify a video')
        return

    # Initialize YOLO model
    YOLO_MODEL = 'yolo11n'
    model = YOLO('{}.pt'.format(YOLO_MODEL))

    # Initialize the tracker
    tracker = bb_tracker()

    # Load the video and process it one frame at a time
    vidcap = cv2.VideoCapture(video_path)
    images_to_process, img = vidcap.read()

    if create_video:
        video = video_editor('out', 'video.mp4', width=int(img.shape[1]/4), height=int(img.shape[0]/4))

    if realtime_display:
        # Display the first image
        fig, ax = plt.subplots()
        imgplot = ax.imshow(img)
        plt.show(block=False)

    num_frames = 0
    start = time()
    
    while images_to_process:

        num_frames+=1

        # Detect objects with YOLO
        yolo_result = model.predict(img, verbose=False)
        
        # Process the detections
        tracker.process_yolo_result(yolo_result)

        # Get all the matched bounding boxes and their properties needed for rendering
        (boxes, confidences, classes, colors) = tracker.get_matches()

        if create_video:
            # Draw matches on the frame and add it to the video
            for box, confidence, type, color in zip(boxes, confidences, classes, colors):
                type = yolo_result[0].names[type]
                img = tracker.drawPred(img, type, confidence, box, color)
            video.add_frame(img)

        if realtime_display:
            # Update the image with the freshly annotated image
            imgplot.set_data(img)
            plt.draw()
            plt.pause(0.001)

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
    # Get a random test image (dataset used is BDD100k)
    video_path = random.choice(os.listdir('videos/test/'))

    main('videos/test/{}'.format(video_path))
