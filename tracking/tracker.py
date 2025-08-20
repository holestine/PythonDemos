import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from math import sqrt, exp
from filterpy.kalman import KalmanFilter

def prevent_division_by_0(value, epsilon=0.01):
    """ Prevents division by 0 and anything else less than epsilon

    Args:
        value (float): A divisor whose size you want to limit
        epsilon (float, optional): The minimum absolute values. Defaults to 0.01.

    Returns:
        float: Epsilon for small values otherwise the value itself
    """
    if value == 0:
        return epsilon
    elif abs(value) < epsilon:
        return np.sign(value) * epsilon
    return value

def convert_bbox_to_x(bbox):
    """Takes a bounding box in the form [x1,y1,x2,y2] and returns x in the form
       [x,y,s,r] where x,y is the centre of the box and s is the scale (i.e., area) 
       and r is the aspect ratio.

    Args:
        bbox (array[4]): The bounding box

    Returns:
        array[4,1]: [x,y s, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    """ Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right

    Args:
        x (array[4]): [x,y,s,r]

    Returns:
        _type_: The bounding box
    """
    center_x     = x[0]
    center_y     = x[1]
    area         = x[2] # w * h
    aspect_ratio = x[3] # w / h

    w = np.sqrt(area * aspect_ratio)
    h = area / w
    return np.array([center_x-w/2., center_y-h/2., center_x+w/2., center_y+h/2.]).reshape((1,4))


class trajectory:
    """ Class that uses the SORT algorithm to track objects """

    # Class variable, incremented to provide a unique id for each trajectory
    current_id = 1

    def __init__(self, box, confidence) -> None:
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

        # Get a unique id
        self.id = trajectory.current_id
        trajectory.current_id = self.id + 1

        # Initialize values for tracking
        self.time_since_update = 0
        self.hit_streak = 0
        self.history = [box]
        self.confidences = [confidence]
        self.initialize_kalman_filter(box)

    def initialize_kalman_filter(self, bbox):
        """
        Initializes a Kalman filter for bounding box predictions.
        """

        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. # Give high uncertainty to the unobservable initial velocities
        self.kf.P        *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_x(bbox)

    def get_prediction(self):
        """ Advances the state vector and returns the predicted bounding box estimate.

        Returns:
            array[4]: The bounding box prediction
        """

        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()

        if(self.time_since_update > 0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x[0:4]).squeeze(0))
        return self.history[-1]

    def update(self, bbox, confidence):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_x(bbox))
        self.confidences.append(confidence)

class bb_detection:
    """ This class is used like a struct to store values associated with individual detections
    """

    def __init__(self, box, confidence) -> None:
        """ Stores the values

        Args:
            box (array): The coordinates of the bounding box
            confidence (float): The confidence of the detection
            type (int): An int identifying the class of the detection
        """
        self.box        = box
        self.confidence = confidence

# Constant indices for YOLO bounding boxes
X_MIN_INDEX = 0
Y_MIN_INDEX = 1
X_MAX_INDEX = 2
Y_MAX_INDEX = 3

class bb_tracker:
    """ A class for tracking bounding boxes in a sequence of images """

    def __init__(self, config) -> None:
        self.confThreshold              = config['confThreshold']
        self.nms_threshold              = config['nms_threshold']
        self.iou_threshold              = config['iou_threshold']
        self.max_missed_detections      = config['max_missed_detections']
        self.min_consecutive_detections = config['min_consecutive_detections']
        self.buffer                     = config['buffer']

        self.trajectories = []

    def process_detections(self, detections, shape):
        """ Adds detection information to tracking history

        Args:
            boxes_xyxy (array): The bounding boxes
            confidences (array): The confidences
            shape (tuple): The shape of the image
        """

        boxes_xyxy  = detections.xyxy
        confidences = detections.confidence

        # Filter out low confidence boxes
        filtered_boxes_xyxy  = boxes_xyxy[confidences >= self.confThreshold]
        filtered_confidences = confidences[confidences >= self.confThreshold]

        # Perform Non Maximum Suppression to remove redundant boxes with low confidence
        indices = cv2.dnn.NMSBoxes(filtered_boxes_xyxy, filtered_confidences, self.confThreshold, self.nms_threshold)

        # Create list of valid detections
        detections = []
        for i in indices:
            detections.append(bb_detection(filtered_boxes_xyxy[i], filtered_confidences[i]))

        # Associate valid detections with the existing trajectories
        self.associate(detections, shape)


    def box_iou(self, box1, box2):
        """ Determines the intersection over union of two bounding boxes

        Args:
            box1 (array): The first box
            box2 (array): The second box

        Returns:
            float: The IOU
        """

        box1_x_min = box1[X_MIN_INDEX] - self.buffer
        box1_y_min = box1[Y_MIN_INDEX] - self.buffer
        box1_x_max = box1[X_MAX_INDEX] + self.buffer
        box1_y_max = box1[Y_MAX_INDEX] + self.buffer
        box2_x_min = box2[X_MIN_INDEX] - self.buffer
        box2_y_min = box2[Y_MIN_INDEX] - self.buffer
        box2_x_max = box2[X_MAX_INDEX] + self.buffer
        box2_y_max = box2[Y_MAX_INDEX] + self.buffer

        max_lhs    = max(box1_x_min, box2_x_min) # The max left hand side
        max_top    = max(box1_y_min, box2_y_min) # The max top
        min_rhs    = min(box1_x_max, box2_x_max) # The min right hand side
        min_bottom = min(box1_y_max, box2_y_max) # The min bottom

        # The intersected area
        inter_area = max(0, min_rhs - max_lhs + 1) * max(0, min_bottom - max_top + 1)

        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1_x_max - box1_x_min + 1) * (box1_y_max- box1_y_min + 1)
        box2_area = (box2[X_MAX_INDEX] - box2[X_MIN_INDEX] + 1) * (box2[Y_MAX_INDEX] - box2[Y_MIN_INDEX] + 1)
        union_area = (box1_area + box2_area) - inter_area
        
        # Compute the IoU
        iou = inter_area/float(union_area)
        return iou
    
    def associate(self, detections, shape):
        """ Associates new detections with the existing trajectories

        Args:
            detections (list): A list of the new detections
            shape (tuple): The height and width of the image
        """

        # Nothing to match so just create new trajectories
        if len(self.trajectories) == 0:
            for d in detections:
                t = trajectory(d.box, d.confidence)
                self.trajectories.append(t)
            return

        # Get the predicted locations for each trajectory
        predictions = [t.get_prediction() for t in self.trajectories]

        # Initialize appropriately dimensioned IOU Matrix
        iou_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float32)

        # Go through all the boxes and store each IOU value
        for i, prediction in enumerate(predictions):
            for j, detection in enumerate([d.box for d in detections]):
                iou_matrix[i][j] = self.cost(prediction, detection, shape)

        # Call the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Go through the matches in the Hungarian Matrix
        for h in hungarian_matrix:
            # If it's under the IOU threshold add a new trajectory
            if iou_matrix[h[0],h[1]] < self.iou_threshold:
                detection = detections[h[1]]
                t = trajectory(detection.box, detection.confidence)
                self.trajectories.append(t)
            else:
                # It's a match, update the trajectory
                self.trajectories[h[0]].update(detections[h[1]].box, detections[h[1]].confidence)

        # Add new trajectories for unmatched detections
        for d, det in enumerate(detections):
            if(d not in hungarian_matrix[:,1]):
                t = trajectory(det.box, det.confidence)
                self.trajectories.append(t)

        # Remove trajectories that haven't been matched for a while
        for t in self.trajectories:
            if t.time_since_update >= self.max_missed_detections:
                self.trajectories.remove(t)

    def get_matches(self):
        """ Returns bounding box information for trajectories with consecutive detections

        Args:
            min_consecutive_detections (int, optional): The minimum number of consecutive detections to be considered a match. 
                                                        Defaults to min_consecutive_detections.

        Returns:
            tuple: The id, bounding boxes and confidences for matched detections
        """

        ids, boxes, confidences = [], [], []
        for t in self.trajectories:
            if t.hit_streak >= self.min_consecutive_detections:
                ids.append(t.id)
                boxes.append(t.history[-1])
                confidences.append(t.confidences[-1])

        return (ids, boxes, confidences)

    def print_matches(self):
        """ Prints all the matches detected in the last update. They can be identified by 
            trajectories that have multiple bounding boxes and zero missed detections.
        """

        print("Matched Detections:")
        for t in self.trajectories:
            if len(t.history) > 1:
                if t.time_since_update == 0:
                    print(t.history[-2])
                    print(t.history[-1])
                    print('\n')

    def print_unmatched_trackers(self):
        """ Prints all the unmatched trackers from the last update. They can be identified by 
            trajectories that have missed detections greater than zero.
        """

        print("Unmatched Trackers:")
        for t in self.trajectories:
            if t.time_since_update > 0:
                print(t.history[-1])

    def print_unmatched_detections(self):
        """ Prints all the unmatched detections from the last update. They can be identified by 
            trajectories that have only one bounding boxes and zero missed detections.
        """
        
        print("Unmatched Detections:")
        for t in self.trajectories:
            if len(t.history) == 1:
                if t.time_since_update == 0:
                    print(t.history[-1])

    def cost(self, prediction, detection, shape, linear_thresh = 10000, exp_thresh = 0.5):
        """ Calculate the IoU cost

        Args:
            prediction (array): The bounding box for the prediction
            detection (array): The bounding box for teh detection
            shape (tuple): The image shape
            linear_thresh (int, optional): Linear threshold value. Defaults to 10000.
            exp_thresh (float, optional): Exponential threshold value. Defaults to 0.5.

        Returns:
            float: The IoU cost
        """

        # IOU COST
        iou_cost = self.box_iou(prediction, detection)
        if (iou_cost < self.iou_threshold):
            return 0
        
        # Sanchez-Matilla et al cost (section 3.1.3 in https://arxiv.org/pdf/1709.03572)
        Q_dist = sqrt(pow(shape[0], 2) + pow(shape[1], 2))
        Q_shape = shape[0] * shape[1]
        distance_term = Q_dist/prevent_division_by_0(sqrt(pow(prediction[X_MIN_INDEX] - detection[X_MIN_INDEX], 2)
                                                        + pow(prediction[Y_MIN_INDEX] - detection[Y_MIN_INDEX],2)))
        
        shape_term = Q_shape/prevent_division_by_0(sqrt(pow(prediction[X_MAX_INDEX] - detection[X_MAX_INDEX], 2)
                                                      + pow(prediction[Y_MAX_INDEX] - detection[Y_MAX_INDEX],2)))
        
        linear_cost = distance_term * shape_term
        if (linear_cost < linear_thresh):
            return 0

        # Yu et al cost (section 3.1.3 in https://arxiv.org/pdf/1709.03572)
        try:
            w1 = 0.5
            w2 = 1.5
            a = (prediction[X_MIN_INDEX] - detection[X_MIN_INDEX]) / prevent_division_by_0(prediction[X_MAX_INDEX])
            a_2 = pow(a,2)
            b = (prediction[Y_MIN_INDEX] - detection[Y_MIN_INDEX]) / prevent_division_by_0(prediction[3])
            b_2 = pow(b,2)
            ab = -1 * (a_2 + b_2) * w1
            c = abs(prediction[Y_MAX_INDEX] - detection[Y_MAX_INDEX])/(prediction[Y_MAX_INDEX] + detection[Y_MAX_INDEX])
            d = abs(prediction[X_MAX_INDEX] - detection[X_MAX_INDEX])/(prediction[X_MAX_INDEX] + detection[X_MAX_INDEX])
            cd = -1 * (c + d) * w2
            exponential_cost = exp(ab) * exp(cd)
            if (exponential_cost < exp_thresh):
                return 0
        except OverflowError:
            print("Overflow exception calculating Yu cost (shouldn't cause an issue).")
        
        return iou_cost
