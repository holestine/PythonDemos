from contextlib import ContextDecorator 
import numpy as np 
import pandas as pd
import time 

class Timer(ContextDecorator): 
    """
    This class is used for timing various stages of execution in a Python script. 
    See the test code below for usage information
    """

    # A dictionary to store the times. This is a class variable 
    # that is accessible to multiple class instantiations allowing
    # multiple things to be timed concurrently
    time_tracker = {}

    def __init__(self, phase):
        """ Adds a new entry to the time tracker when necessary

        Args:
            phase (string): The name of the phase
        """
        
        if isinstance(phase, str):
            self.phase = phase  
            if not phase in Timer.time_tracker:
                # Add list to dictionary entry to store execution times
                Timer.time_tracker[self.phase] = {'times':[]}  

    def __enter__(self):
        """ Stores the start time
        """
        self.start = time.time()  

    def __exit__(self, *args):  
        """ Computes and stores the execution time
        """
        self.end = time.time()  
        Timer.time_tracker[self.phase]['times'].append(self.end - self.start)  

    @classmethod
    def report_phase(self, phase, times, print_all_times=False):
        """ Prints out metrics for the execution phase.

        Args:
            phase (string): The name of the phase.
            times (array): The execution times.
            print_all_times (bool, optional): A flag indicating if all execution times should be displayed. Defaults to False.
        """
        print(f'{phase} ran {len(times)} times')
        
        if print_all_times:
            print(f'\tTimes: {times}' ) 
        
        print(f'\tMin time was {1000 * np.min(times)} at index {times.index(np.min(times))}')  
        print(f'\tMax time was {1000 * np.max(times)} at index {times.index(np.max(times))}') 
        print(f'\tAverage time was {1000 * np.mean(times)} ms')  
        print(f'\tTotal time was {1000 * np.sum(times)} ms\n')

    @classmethod
    def report(self, filename=None):
        """ Prints out the metrics for each execution phase and optionally saves it to a file.

        Args:
            filename (string, optional): Location to store the metrics. Defaults to None.
        """
        if filename:
            data = []

        for phase, times in Timer.time_tracker.items():  
            Timer.report_phase(phase, times['times'])
            if filename:
                data.append([phase, np.min(times['times']), np.max(times['times']), np.mean(times['times']), np.sum(times['times'])])

        if filename:
            pd.DataFrame(data, columns=['Name', 'Min', 'Max', 'Mean', 'Total']).to_csv(filename, index=False)

    @classmethod
    def reset(self):
        """ Clears data from the current tracker.
        """
        Timer.time_tracker.clear()
        self.phase = None

def test_phases():
    """ Prints metric for consecutive phases.
    """
    phase = 'PHASE ONE'
    for _ in range(1):
        with Timer(phase):
            time.sleep(.1)

    phase = 'PHASE TWO'
    for _ in range(2):
        with Timer(phase):
            time.sleep(.2)

    phase = 'PHASE THREE'
    for _ in range(3):
        with Timer(phase):
            time.sleep(.3)

    Timer.report()

def test_csv():
    """ Saves the current metrics to a CSV
    """
    Timer.report('timer.csv')

def test_reset():
    """ Clears current data, starts a new phase and displays the metrics
    """
    Timer.reset()

    phase = 'PHASE FOUR'
    for _ in range(4):
        with Timer(phase):
            time.sleep(.4)

    Timer.report()
    Timer.reset()

def test_nested():
    """ Creates concurrent times and prints the metrics.
    """
    
    with Timer('OUTSIDE'):
        for _ in range(2):
            with Timer('INSIDE'):
                time.sleep(.1)

    Timer.report()

if __name__ == '__main__':
    """ A few tests to show usage and make sure of proper functionality. 
    """
    test_phases()
    test_csv()
    test_reset()
    test_nested()
