
class FeatureDetector:
    def check(self, frame):
        '''
        This method receives a frame and returns the probability that a feature
        exists on this frame

        :param frame: numpy array containing the frame in BGR
        :return: probability between 0 and 1
        '''
        raise NotImplementedError()
