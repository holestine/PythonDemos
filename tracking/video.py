import os, cv2

class video_editor:

    def __init__(self, out_dir, filename, fps=30, width=1200, height=800, is_rgb=True) -> None:
        '''
        Creates a video 
    
        Parameters
        ----------
        out_dir : str
            The directory to store the video
        filename : str
            The name of the video
        fps : int
            The frames per second
        width : int
            The video width
        height : int
            The video height
        is_rgb : bool
            Flag to specify whether or not the video will be in color
        '''

        # Make sure directory is there
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Construct the path
        video_path = os.path.join(out_dir, filename)

        # Store video size
        self.width, self.height = width, height

        # Construct the video writer
        self.out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.width, self.height), is_rgb)

    def add_frame(self, img) -> None:
        '''
        Adds a single frame to the video 
    
        Parameters
        ----------
        img : array(int)
            The image
        '''
        img = cv2.resize(img, (self.width, self.height))
        self.out.write(img)

    def save_video(self) -> None:
        '''
        Saves the video
        '''

        self.out.release()
