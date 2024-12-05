import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from math import sqrt, exp

def prevent_division_by_0(value, epsilon=0.01):
    """ Prevents division by 0 and anything else less than epsilon

    Args:
        value (float): A divisor whose size you want to limit
        epsilon (float, optional): The minimum absolute values. Defaults to 0.01.

    Returns:
        float: Epsilon for small values otherwise the value itself
    """
    
    if abs(value) < epsilon:
        return np.sign(value) * epsilon
    return value

class trajectory:

    # Shared class variable
    current_id = 1

    def __init__(self, box, confidence, type) -> None:
        """
        Initializes parameters for a new trajectory

        Parameters
        ----------
        box : array[4]
            The bounding box for the current detection
        confidence : float
            A value between 0 and 1 indicating confidence of the detection
        type : int
            A value that indicates the type of the detection (e.g., person, car ...)
        """

        # Get a unique id and color
        self.id = trajectory.current_id
        trajectory.current_id = self.id + 1
        self.color = self.id_to_color(self.id)

        # Get the type
        self.type = type

        # Initialize values for tracking 
        self.boxes = [box]
        self.confidences = [confidence]
        self.missed_detections = 0
        self.consecutive_detections = 0

    def id_to_color(self, id):
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

class yolo_detection:

    def __init__(self, box, confidence, type) -> None:
        """ This class is used like a struct to store values associated with individual detections

        Args:
            box (array): The coordinates of the bounding box
            confidence (float): The confidence of the detection
            type (int): An int identifying the class of the detection
        """

        self.box        = box
        self.confidence = confidence
        self.type       = type

X_MIN_INDEX = 0
Y_MIN_INDEX = 1
X_MAX_INDEX = 2
Y_MAX_INDEX = 3

class bb_tracker:
    """ A class for tracking bounding boxes in a sequence of images
    """
    
    # Constants
    confThreshold = 0.2
    nmsThreshold = 0.8
    iou_threshold = 0.3
    max_missed_detections = 30
    min_consecutive_detections = 3

    def __init__(self) -> None:
        
        self.trajectories = []

    def process_yolo_result(self, yolo_result):
        '''
        Incorporates the results from YOLO into the current trajectories
    
        Parameters
        ----------
        yolo_result : object
            The decoded results from the YOLO model
        '''

        # Get the bounding box locations and the associated classes and confidences
        boxes_xyxy  = yolo_result[0].boxes.xyxy.cpu().numpy()
        confidences = yolo_result[0].boxes.conf.cpu().numpy()
        classes     = yolo_result[0].boxes.cls.cpu().numpy()

        # Filter out low confidence boxes
        filtered_boxes_xyxy  = boxes_xyxy[confidences > bb_tracker.confThreshold]
        filtered_confidences = confidences[confidences > bb_tracker.confThreshold]
        filtered_classes     = classes[confidences > bb_tracker.confThreshold].astype(int)

        # Perform Non Maximum Suppression to remove redundant boxes with low confidence
        indices = cv2.dnn.NMSBoxes(filtered_boxes_xyxy, filtered_confidences, bb_tracker.confThreshold, bb_tracker.nmsThreshold)

        # Create list of valid detections
        detections = []
        for i in indices:
            detection = yolo_detection(filtered_boxes_xyxy[i], filtered_confidences[i], filtered_classes[i])
            detections.append(detection)

        # Associate valid detections with the existing trajectories
        self.associate(detections, yolo_result[0].orig_shape)

    def box_iou(self, box1, box2):
        '''
        Determines the intersection over union of two bounding boxes
    
        Parameters
        ----------
        box1 : array[4]
            The first box
        box2 : array[4]
            The second box
        '''

        max_lhs    = max(box1[X_MIN_INDEX], box2[X_MIN_INDEX]) # The max left hand side
        max_top    = max(box1[Y_MIN_INDEX], box2[Y_MIN_INDEX]) # The max top
        min_rhs    = min(box1[X_MAX_INDEX], box2[X_MAX_INDEX]) # The min right hand side
        min_bottom = min(box1[Y_MAX_INDEX], box2[Y_MAX_INDEX]) # The min bottom

        # The intersected area
        inter_area = max(0, min_rhs - max_lhs + 1) * max(0, min_bottom - max_top + 1) 

        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1[X_MAX_INDEX] - box1[X_MIN_INDEX] + 1) * (box1[Y_MAX_INDEX] - box1[Y_MIN_INDEX] + 1)
        box2_area = (box2[X_MAX_INDEX] - box2[X_MIN_INDEX] + 1) * (box2[Y_MAX_INDEX] - box2[Y_MIN_INDEX] + 1)
        union_area = (box1_area + box2_area) - inter_area
        
        # Compute the IoU
        iou = inter_area/float(union_area)
        return iou

    def drawPred(self, frame, type, conf, box, color=(255, 0, 0)):
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
        label = "{}: {:.2f}".format(type, conf)

        # Display the label at the top of the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 3
        color = (255,255,255)
        labelSize, _ = cv2.getTextSize(label, font, scale, thickness)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), font, scale, color, thickness, cv2.LINE_AA)

        return frame
    
    def associate(self, detections, shape):
        '''
        Associates new detections with the existing trajectories
    
        Parameters
        ----------
        detections : list
            A list of the new detections
        '''

        # Nothing to match so just create new trajectories
        if len(self.trajectories) == 0:
            for detection in detections:
                t = trajectory(detection.box, detection.confidence, detection.type)
                self.trajectories.append(t)
            return

        # Get the last know location for each trajectory
        old_boxes = [t.boxes[-1] for t in self.trajectories]
                         
        # Get the location of all the new detections
        new_boxes = [detection.box for detection in detections]

        # Define an IOU Matrix with dimensions old_boxes x new_boxes
        iou_matrix = np.zeros((len(old_boxes), len(new_boxes)), dtype=np.float32)

        # Go through all the boxes and store each IOU value
        for i, old_box in enumerate(old_boxes):
            for j, new_box in enumerate(new_boxes):
                iou_matrix[i][j] = self.hungarian_cost(old_box, new_box, shape)

        # Call the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Go through the matches in the Hungarian Matrix 
        for h in hungarian_matrix:
            # If it's under the IOU threshold increment the missed detections in the tracker and add a new trajectory
            if iou_matrix[h[0],h[1]] < bb_tracker.iou_threshold or self.trajectories[h[0]].type != detections[h[1]].type:
                self.trajectories[h[0]].missed_detections += 1
                detection = detections[h[1]]
                t = trajectory(detection.box, detection.confidence, detection.type)
                self.trajectories.append(t)
            # Else, it's a match, add the box to the trajectory
            else:
                self.trajectories[h[0]].boxes.append(detections[h[1]].box)
                self.trajectories[h[0]].consecutive_detections = max(1, self.trajectories[h[0]].consecutive_detections+1)
                self.trajectories[h[0]].missed_detections = 0
                self.trajectories[h[0]].confidences.append(detections[h[1]].confidence)
               
        # Add new trajectories for unmatched detections
        new_boxes = [detection.box for detection in detections]
        for d, det in enumerate(detections):
            if(d not in hungarian_matrix[:,1]):
                t = trajectory(det.box, det.confidence, det.type)
                self.trajectories.append(t)
        
        # Keep track of missed and consecutive detections, remove trajectories that have not been matched for a while
        for t in self.trajectories:
            if len(new_boxes) == 0 or not np.any(np.all(t.boxes[-1] == new_boxes, axis=1)):
                if t.missed_detections >= bb_tracker.max_missed_detections:
                    self.trajectories.remove(t)
                else:
                    t.missed_detections += 1
                    t.consecutive_detections = 0

    def get_matches(self, min_consecutive_detections=min_consecutive_detections):
        """ Returns bounding box information for trajectories with consecutive detections

        Args:
            min_consecutive_detections (int, optional): The minimum number of consecutive detections to be considered a match. Defaults to min_consecutive_detections.

        Returns:
            tuple: The bounding boxes, their colors, the confidences and classes for matched detections
        """

        boxes, confidences, classes, colors = [], [], [], []
        for t in self.trajectories:
            if t.consecutive_detections >= min_consecutive_detections:
                boxes.append(t.boxes[-1])
                confidences.append(t.confidences[-1])
                classes.append(t.type)
                colors.append(t.color)

        return (boxes, confidences, classes, colors)

    def print_matches(self):
        """ Prints all the matches detected in the last update. They can be identified by trajectories that have multiple bounding boxes and zero missed detections.
        """

        print("Matched Detections:")
        for t in self.trajectories:
            if len(t.boxes) > 1:
                if t.missed_detections == 0:
                    print(t.boxes[-2])
                    print(t.boxes[-1])
                    print('\n')

    def print_unmatched_trackers(self):
        """ Prints all the unmatched trackers from the last update. They can be identified by trajectories that have missed detections greater than zero.
        """

        print("Unmatched Trackers:")
        for t in self.trajectories:
            if t.missed_detections > 0:
                print(t.boxes[-1])

    def print_unmatched_detections(self):
        """ Prints all the unmatched detections from the last update. They can be identified by trajectories that only one bounding boxes and zero missed detections.
        """
        
        print("Unmatched Detections:")
        for t in self.trajectories:
            if len(t.boxes) == 1:
                if t.missed_detections == 0:
                    print(t.boxes[-1])

    def hungarian_cost(self, old_box, new_box, shape, linear_thresh = 10000, exp_thresh = 0.5):

        # IOU COST
        iou_cost = self.box_iou(old_box, new_box)
        if (iou_cost < bb_tracker.iou_threshold):
            return 0
        
        ### Sanchez-Matilla et al COST
        Q_dist = sqrt(pow(shape[0], 2) + pow(shape[1], 2))
        Q_shape = shape[0] * shape[1]
        distance_term = Q_dist/prevent_division_by_0(sqrt(pow(old_box[X_MIN_INDEX] - new_box[X_MIN_INDEX], 2)
                                                          + pow(old_box[Y_MIN_INDEX] - new_box[Y_MIN_INDEX],2)))
        
        shape_term = Q_shape/prevent_division_by_0(sqrt(pow(old_box[X_MAX_INDEX] - new_box[X_MAX_INDEX], 2)
                                                        + pow(old_box[Y_MAX_INDEX] - new_box[Y_MAX_INDEX],2)))
        
        linear_cost = distance_term * shape_term
        if (linear_cost < linear_thresh):
            return 0

        ## YUL et al COST
        w1 = 0.5
        w2 = 1.5
        a = (old_box[X_MIN_INDEX] - new_box[X_MIN_INDEX]) / prevent_division_by_0(old_box[X_MAX_INDEX])
        a_2 = pow(a,2)
        b = (old_box[Y_MIN_INDEX] - new_box[Y_MIN_INDEX])/prevent_division_by_0(old_box[3])
        b_2 = pow(b,2)
        ab = -1 * (a_2 + b_2) * w1
        c = abs(old_box[Y_MAX_INDEX] - new_box[Y_MAX_INDEX])/(old_box[Y_MAX_INDEX] + new_box[Y_MAX_INDEX])
        d = abs(old_box[X_MAX_INDEX] - new_box[X_MAX_INDEX])/(old_box[X_MAX_INDEX] + new_box[X_MAX_INDEX])
        cd = -1 * (c + d) * w2
        exponential_cost = exp(ab) * exp(cd)

        if (exponential_cost < exp_thresh):
            return 0
        
        return iou_cost
        