import os
import os.path as path
import json

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import scipy.misc as smisc
from keras.models import model_from_json
from keras.optimizers import Adadelta

from features.feature_detector import FeatureDetector


class HeroDetector(FeatureDetector):
    '''
    Identify which hero a player plays.
    '''

    def __init__(self,
                 cnn_struct_file=path.join('..', 'service_files', 'service_hero_files', 'cnn928_24.json'),
                 cnn_weights_file=path.join('..', 'service_files', 'service_hero_files', 'cnn928_24.h5'),
                 mask_file=path.join('..', 'service_files', 'service_hero_files', 'mask.png'),
                 heroes_names_file=path.join('..', 'service_files', 'service_hero_files', 'heroes_names.json'),
                 coords=(500, 500, 200),
                 roi_border=0,
                 gray=False):
        '''
        Initialize a recognizer with a defined neural network classifier an coordinates for ROI.
        :param cnn_struct_file: json file with the convolutional neural network architecture
        :param cnn_weights_file: h5 file with neural network weights
        :param mask_file: image file with a mask
        :param heroes_names_file: json file with the list of heroes names
        :param coords: (int, int, int) - coordinates of left upper corner of ROI and window size of hero picture
                       (y, x, window)
        :return: None
        '''
        self.gray = gray

        # read self mask
        self.mask = cv2.imread(mask_file)

        # load neural network classifier
        self.pred_size = int(cnn_struct_file[-7:-5])
        json_file = open(cnn_struct_file)
        loaded_json_model = json_file.read()
        json_file.close()
        self.cnn = model_from_json(loaded_json_model)
        self.cnn.load_weights(cnn_weights_file)

        self.cnn.compile(loss='categorical_crossentropy',
                         optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0, clipnorm=0.5),
                         metrics=['accuracy'])
        nb_channels = 1 if self.gray else 3
        self.cnn.predict(np.zeros((1, self.pred_size, self.pred_size, nb_channels)))

        # read heroes names dictionary
        with open(heroes_names_file) as f:
            self.heroes_names = json.load(f)

        # set parameters for roi
        self.coords = coords
        self.roi_border = roi_border

        #####################
        self.probs = np.array([])

    def _hough_detect(self, img, dp=1, min_dist=300, param1=20, param2=65,
                      min_radius=12, max_radius=80):
        '''
        Detect circles with Hough transformation detector
        :param img: numpy array - an image to detect circles
        :param dp: int - inverse ration of the accumulator resolution to the image resolution
        :param min_dist: int - minimum distance between the centers of the detected circles
        :param param1: int - the higher threshold of the two passed to the Canny() edge detector
        :param param2: int - the accumulator threshold for the circle centers at the detection stage
        :param min_radius: int - minimum circle radius
        :param max_radius: int - maximum circle radius
        :return: list of int - list of coordinates [x, y, R] of detected circles
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, min_dist,
                                  param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
        return circles

    def check(self, frame):
        '''
        Identify which hero is contained in frame.
        :param frame: numpy array - a frame to extract hero name
        :return: (float, str) - a probability of hero picture, a name of hero picture
        '''
        # detect ROI with hero picture
        x_start = int(frame.shape[1]/2) - self.coords[0]
        y_start = frame.shape[0] - self.coords[2]
        roi_start = frame[y_start:, x_start:x_start+self.coords[1]]
        roi_coords = self._hough_detect(roi_start)

        # return None if hero contour was not detected
        if roi_coords is None:
            return 0, 'None'

        # extract ROI coordinates
        x_add, y_add, radius = np.uint16(roi_coords)[0][0]
        picture = frame[y_start+y_add-radius-self.roi_border: y_start+y_add+radius+self.roi_border,
                        x_start+x_add-radius-self.roi_border: x_start+x_add+radius+self.roi_border]

        # add mask
        if self.mask.shape != picture.shape:
            mask = smisc.imresize(self.mask, (picture.shape[0], picture.shape[1]))
            mask.astype(np.bool)
        else:
            mask = self.mask
        bool_mask = mask > 50
        masked_picture = bool_mask * picture

        # fit to a size predicted by the neural network
        masked_picture = smisc.imresize(masked_picture, (self.pred_size, self.pred_size))

        # convert a picture to gray
        if self.gray:
            masked_picture = cv2.cvtColor(masked_picture, cv2.COLOR_BGR2GRAY)
            masked_picture = masked_picture.reshape((masked_picture.shape[0], masked_picture.shape[1], 1))
        # normalize a picture
        masked_picture = np.array([masked_picture], dtype=np.float)
        masked_picture = cv2.normalize(masked_picture, masked_picture, norm_type=cv2.NORM_MINMAX)

        # predict hero
        hero_probs = self.cnn.predict(masked_picture)
        hero_prob = np.max(hero_probs[0])
        hero_name = self.heroes_names[str(np.argmax(hero_probs[0]))]
        ##############
        self.probs = hero_probs

        return hero_prob, hero_name
