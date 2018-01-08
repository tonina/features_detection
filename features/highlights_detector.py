import os
import os.path as path
import re

import cv2
import joblib
import numpy as np
import pytesseract
from PIL import Image
from difflib import SequenceMatcher
import scipy.ndimage as ndimage
import scipy.misc as smisc
from sklearn.preprocessing import MinMaxScaler

from features.feature_detector import FeatureDetector


def read_text(bin_image):
    '''
    Recognize text from binary image with tesseract.
    :param bin_image: numpy array - binary image
    :return: str
    '''
    res_image = Image.fromarray(bin_image)
    text = pytesseract.image_to_string(res_image, config=tessdata_dir_config)
    return text


def insert_symbol(s, idx, symbol=' '):
    '''
    Insert symbol or substring in string by index.
    :param s: str
    :param idx: int - index
    :param symbol: str - char or substring
    :return: str
    '''
    return s[:idx] + symbol + s[idx:]


class Rectangle(object):
    '''
    Class describes a rectangle by coordinates.
    '''
    def __init__(self, x1, y1, x2, y2):
        '''
        Initialize a rectangle by coordinates.
        :param x1: x coordinate of upper left corner
        :param y1: y coordinate of upper left corner
        :param x2: x coordinate of lower right corner
        :param y2: y coordinate of lower right corner
        :return: None
        '''
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class HighlightsDetector(FeatureDetector):
    '''
    Class for detection and recognition highlighs - the pairs of player name and icon.

    Parameters:
    Parameters of highlighs roi:
        corner_coords: coordinates of upper right corner for looking for highlights (default=(400, 400))

    Icon detection and recognition parameters:
        icon_dscale: downscale factor of icons templates and images for icon detection and recognition (default=2)
        start_scale: start value for rescaling icon templates (default=0)
        end_scale: end value for rescaling icon templates (default=12)
        close_dist_icon: maximum distance for join icon regions (default=4)
        thresh_icon: threshold for select icon regions by template mathcing (default=0.8)

    Text detection parameters:
        blue: lower and upper values for mask filtered blue (default=[[90, 120, 230], [140, 160, 255]])
        orange: lower and upper values for mask filtered orange (default=[[230, 140, 80], [255, 175, 125]])
        white: lower and upper values for mask filtered white (default=[[140, 180, 190], [255, 255, 255]])
        min_roi_area: minimum area of text roi (default=500)
        max_roi_area: maximum area of text roi (default=15000)
        sides_ratio: sides ration for text roi (default=1.0)
        close_dist_text: maximum distance for join two nearby text roi (default=15)
        kernel_size: size of rectangle kernel for closing morphological operation (default=(5, 5))
        text_icon_distance: maximum distance between icon and text (default=75)

    Char classification parameters:
        dsize: size of recognized chars by pretrained knn classifier (default=(32, 32))
               to modify this value corresponding text classifier should be saved to file
               the name of which is pointed by text_classifier
        text_scale: upscale factor for text splitting and recognition (default=(2, 2))

    Players names extraction parameters
        start_y: y start coordinate of roi with players names (default=224)
        end_y: y end coordinate of roi with players names (default=486)
        step_y: y step between players names (default=36)
        hight: y high of player name (default=25)
        start_x: x start coordinate of roi with players names (default=1000)
        end_x: x end coordinate of roi with players names (default=1200)
        base: base resolution for scale (default=720)
        scale: upscale factor for tesseract recognition (default=4)
        thresh_names: threshold for binarization (default=175)

    Parameters setting example:
        hl_detector = HighlightsDetector()
        hl_detector.start_scale = -4
        hl_detector.end_scale = 20
    '''
    def __init__(self,
                 template_dir=path.join('..', 'service_files', 'service_highlights_files', 'templates',),
                 text_classifier=path.join('..', 'service_files', 'service_highlights_files', 'text_knn.pkl'),
                 char_labels=path.join('..', 'service_files', 'service_highlights_files', 'chars.pkl')
                 ):
        '''
        Initialize parameters of detector including a classifier of icon.
        :param template_dir: str - name of directory with templates
        '''
        # parameters of highlighs roi
        self.corner_coords = (400, 400)

        # icon detection and recognition parameters
        self.icon_dscale = 2
        self.start_scale = 0
        self.end_scale = 12
        self.close_dist_icon = 4
        self.thresh_icon = 0.8

        # text detection parameters
        self.blue = np.array([[90, 120, 230], [140, 160, 255]])
        self.orange = np.array([[230, 140, 80], [255, 175, 125]])
        self.white = np.array([[140, 180, 190], [255, 255, 255]])
        self.min_roi_area = 200
        self.max_roi_area = 15000
        self.sides_ratio = 1.0
        self.close_dist_text = 20
        self.kernel_size = (5, 5)
        self.text_icon_distance = 75

        # char classification parameters
        self.dsize = (32, 32)
        self.text_scale = (2, 2)

        # players names extraction parameters
        self.start_y = 224
        self.end_y = 486
        self.step_y = 36
        self.hight = 25
        self.start_x = 1000
        self.end_x = 1200
        self.base = 720
        self.scale = 4
        self.thresh_names = 175

        # initialize area for highlights detection
        self.corner = None
        cv2.inRange(np.zeros(1), 0, 0)

        # initialize kernel for closing operation
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)

        # read templates title for icon detection
        self.template_titles = [str(f)[:-4] for f in os.listdir(template_dir)]

        # read, downscale and convert to gray template images
        self.templates = [ndimage.zoom(cv2.imread(path.join(template_dir, f), 0),
                                       (1 / self.icon_dscale, 1 / self.icon_dscale))
                          for f in os.listdir(template_dir)]

        # load text classifier and set parameters
        with open(text_classifier, 'rb') as fp:
            self.clf = joblib.load(fp)
        self.scaler = MinMaxScaler()

        # load char labels
        with open(char_labels, 'rb') as fp:
            self.char_labels = joblib.load(fp)

        # initialize players names list
        self.names = ['YOU', ]

    def _detect_text_roi(self):
        '''
        Detect regions with names, left part.
        :return: list of text roi
        '''
        # apply a colour mask to corner image
        blue_mask = cv2.inRange(self.corner, self.blue[0], self.blue[1])
        orange_mask = cv2.inRange(self.corner, self.orange[0], self.orange[1])
        white_mask = cv2.inRange(self.corner, self.white[0], self.white[1])
        mask = cv2.bitwise_or(blue_mask, orange_mask)
        mask = cv2.bitwise_or(mask, white_mask)

        # find contours of selected coloured regions
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        im, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # filter regions by area and sides ratio
        regions = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            box = cv2.minAreaRect(contour)
            if self.min_roi_area < cv2.contourArea(contour) < self.max_roi_area\
               and (0 <= abs(box[2]) <= 15 or 75 <= abs(box[2]) <= 90):
                left = np.min(contour, axis=0)[0]  # x, y of upper left point of roi
                right = np.max(contour, axis=0)[0]  # x, y of lower right point of roi
                if (right[0] - left[0])/(right[1] - left[1]) > self.sides_ratio:
                    regions.append(Rectangle(left[0], left[1], right[0], right[1]))

        # combine regions nearby on an axis x
        add_regions = []
        for region in regions:
            for test_region in regions:
                if region is not test_region:
                    if (abs(region.x2 - test_region.x1) < self.close_dist_text or
                       abs(region.x2 - test_region.x1) < self.close_dist_text) and \
                       abs(region.y1 - test_region.y1) < self.close_dist_text and \
                       abs(region.y2 - test_region.y2) < self.close_dist_text:
                        add_regions.append(Rectangle(region.x1, min(region.y1, test_region.y1,),
                                                     test_region.x2, max(region.y1, test_region.y2)))
                        regions.remove(region)
                        regions.remove(test_region)
        regions = regions + add_regions

        # remove enclosed regions
        for region in regions:
            for test_region in regions:
                if region is not test_region:
                    if region.x1 <= test_region.x1 and region.y1 <= test_region.y1 and \
                       region.x2 >= test_region.x2 and region.y2 >= test_region.y2:
                        regions.remove(test_region)
        return regions

    def _split_text(self, image):
        '''
        Split word on text image into character images.
        :param image: numpy array - text image for recognition
        :return: list of numpy arrays - characters images for recognition
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = ndimage.zoom(gray, self.text_scale)

        # find mser regions
        mser = cv2.MSER_create()
        regions = mser.detectRegions(gray)[0]
        roi = np.array([[np.min(reg, axis=0), np.max(reg, axis=0)] for reg in regions])
        # select region with width less than high
        roi = roi[np.where(((roi[:, 1, 0] - roi[:, 0, 0])/(roi[:, 1, 1] - roi[:, 0, 1])) < 1)]

        # select one roi per letter from repeating
        char_roi = list()
        chars = list()
        x = 0
        while x <= gray.shape[1]:
            rest_roi = roi[np.where(roi[:, 0, 0] > x)]
            if rest_roi.size:
                rect = rest_roi[np.where(rest_roi[:, 0, 0] == np.min(rest_roi[:, 0, 0]))][0]
                x = rect[1][0]
                if rect[0][0] != rect[1][0] and rect[1][0] != rect[1][1]:
                    char_roi.append(rect)
                    chars.append(gray[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]])
            else:
                break
        distance = [char_roi[i+1][0][0] - char_roi[i][1][0] for i in range(len(char_roi) - 1)]
        spaces_idx = list()
        for i in range(len(char_roi) - 1):
                if distance[i] >= np.mean(distance) + 1:
                    spaces_idx.append(i)
        return chars, spaces_idx

    def _recognize_text(self, text_image):
        '''
        Recognize text on the image.
        :param text_image: numpy array - image with text
        :return: str - recognized text
        '''
        chars, spaces_idx = self._split_text(text_image)
        chars_list = list()
        text = ''

        # process and recognize characters
        for char in chars:
            char = smisc.imresize(char, self.dsize)
            char = self.scaler.fit_transform(char.astype(float))
            char = char.reshape(char.shape[0] * char.shape[1])
            chars_list.append(char)
        chars_arr = np.array(chars_list)

        # predict characters
        if chars_arr.size:
            pred_labels = self.clf.predict(chars_arr)
            text = text.join([self.char_labels[label] for label in pred_labels])

        # insert spaces
        for idx in spaces_idx:
            text = insert_symbol(text, idx)

        return text

    def _detect_icons(self):
        '''
        Detect and recognize icons.
        :return: dict with icon titles and icon roi
        '''
        icons = {}
        # initialize start and end number of pixel for template resizing
        start_scale = self.start_scale
        end_scale = self.end_scale
        # convert corner image to gray and downscale
        img_gray = cv2.cv2.cvtColor(ndimage.zoom(self.corner, (1 / self.icon_dscale, 1 / self.icon_dscale, 1)),
                                    cv2.COLOR_BGR2GRAY)
        # look for template matches
        for i in range(len(self.templates)):
            # scale template
            for add in range(start_scale, end_scale, 2):
                template = smisc.imresize(self.templates[i],
                                          (self.templates[i].shape[0]+add, self.templates[i].shape[1]+add))
                width, height = template.shape[::-1]
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.thresh_icon)
                points = [p for p in zip(*loc[::-1])]
                # remove repeating icon roi
                if points:
                    for check in range(2):
                        for p in points:
                            for test_p in points:
                                if p != test_p:
                                    if abs(p[0] - test_p[0]) <= self.close_dist_icon \
                                       and abs(p[1] - test_p[1]) <= self.close_dist_icon:
                                        points.remove(test_p)
                    # compute coordinates of roi
                    points = [(self.icon_dscale * p[0], self.icon_dscale * p[1],
                               self.icon_dscale * (p[0] + width), self.icon_dscale * (p[1] + height))
                              for p in points]
                    # add coordinates to icon dictionary with icon title
                    icons[self.template_titles[i]] = points
                    # change end and start number of pixels
                    if add >= 0:
                        start_scale = add - 2
                        end_scale = add + 4
                    else:
                        start_scale = - 2
                        end_scale = 4
                    # bread scaling template if template is found
                    break
        return icons

    def check(self, frame):
        '''
        Recognize highlights (a player name and an icon) in the frame.
        :param frame: numpy array - frame for recognition
        :return: list of tuple - pairs of player name and icon title
        '''
        highlighs = []
        # select upper corner with highlights
        x_start = frame.shape[1] - self.corner_coords[0]
        y_end = self.corner_coords[1]
        self.corner = frame[:y_end, x_start:]

        # detect text roi
        text_roi = self._detect_text_roi()
        # detect and recognize icons
        icons = self._detect_icons()

        # put in compliance icon and text roi and detect player name
        # look over icon regions
        for key in icons:
            for p in icons[key]:
                # look over text regions
                for roi in text_roi:
                    if (p[1] < roi.y1 and roi.y2 < p[3]
                       or abs(roi.y1 - p[1]) <= self.close_dist_text and abs(roi.y2 - p[3]) <=self.close_dist_text) \
                       and roi.x1 < p[0]\
                       and abs(p[0] - roi.x2) < self.text_icon_distance:
                        # extract text
                        text_image = self.corner[roi.y1: roi.y2, roi.x1:p[0]]
                        text = self._recognize_text(text_image)
                        # compare with text from self.names
                        similarity = [SequenceMatcher(None, text, name).ratio() for name in self.names]
                        if max(similarity) > 0.5:
                            text = self.names[similarity.index(max(similarity))]
                        highlighs.append((key, text))
        return highlighs

    def extract_names(self, frame):
        '''
        Extract players names from a frame in the beginning of the game, save to self.names.
        :param frame: numpy array - frame for recognition
        :return: None
        '''
        # scale resolution factor
        k = frame.shape[0]/self.base
        names = list()
        # look over regions with players names
        for s in range(self.start_y, self.end_y, self.step_y):
            # cut region with player name
            name_area = frame[int(k*s):int(k*(s+self.hight)), int(k*self.start_x):int(k*self.end_x)]
            # extract green channel and threshold
            b, g, r = cv2.split(name_area)
            g = ndimage.zoom(g, self.scale)
            ret, th = cv2.threshold(g, self.thresh_names, 255, cv2.THRESH_BINARY_INV)
            text = read_text(th)
            if not text:
                ret, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                text = read_text(th)
            names.append(text)

        # filter words with more than one characters before new line
        for item in names:
            name = ' '.join(re.findall(r'\w{2,}', item[:item.find('\n')]))
            if name and 'Serv' not in name:
                self.names.append(name.upper())
