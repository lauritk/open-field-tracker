import cv2 as cv
import numpy as np

green = (70, 121, 0)
pink = (255, 159, 228)
yellow = (0, 123, 237)
blue = (222, 154, 2)
red = (51, 0, 155)


# TODO: Split classes to separate files


class Frame:
    
    def __init__(self, img=None, num=None, time=None):
        self.img = img
        if img is not None:
            self.img_org = img.copy()
        else:
            self.img_org = None
        self.num = num
        self.time = time
    
    def size(self):
        return self.img.shape[1], self.img.shape[0]


class Capture:
    
    def __init__(self, source):
        self.source = source
        self.capture = cv.VideoCapture(self.source)
        self.frame = Frame()
        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        # Frame count from meta is not to be trusted?
        # self.total_frames = self.capture.get(cv.CAP_PROP_FRAME_COUNT)
        self.capture.set(cv.CAP_PROP_POS_AVI_RATIO, 1)
        self.total_frames = int(self.capture.get(cv.CAP_PROP_POS_FRAMES))
        self.capture.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
        self.capture.open(self.source) # Fixes file closed bug
        self.duration_ms = self.total_frames / self.fps * 1000

    def get_fps(self):
        return self.fps

    def get_frame_count(self):
        return self.total_frames
    
    def get_duration_ms(self):
        return self.duration_ms
    
    def capture_frame(self):
        try:
            ret, img = self.capture.read()
            if ret:
                self.frame.img = img
                self.frame.img_org = img.copy()
                self.frame.num = int(self.capture.get(cv.CAP_PROP_POS_FRAMES))
                self.frame.time = self.capture.get(cv.CAP_PROP_POS_MSEC)
                return self.frame
            else:
                raise IOError('Frame could not be read!')
        except IOError:
            self.release_capture()
    
    def is_open(self):
        return self.capture.isOpened()
    
    def release_capture(self):
        print('Releasing video source...')
        self.capture.release()
    
    def show(self, window='frame', img=None):
        if img is not None:
            cv.imshow(window, img)
        else:
            cv.imshow(window, self.frame.img)
    
    def duration_in_s(self):
        return self.total_frames / self.fps
    
    def show_img(self, window, img=None):
        while True:
            if img is None:
                cv.imshow(window, self.frame.img)
            else:
                cv.imshow(window, img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()


class Transformation:
    
    def __init__(self, frame=Frame(), corners=None,
                 transfor_matrix=None, f_dim=None, pad=100, width=None, height=None):
        self.frame = frame
        self.corners = corners
        self.f_dim = f_dim
        if corners is not None and width is None:
            self.corners = [x[0] for x in self.corners]
            self.width = self.corners[3][0] - self.corners[0][0]
            self.height = self.corners[3][1] - self.corners[0][1]
        else:
            self.width = width
            self.height = height
        self.pad = pad
        self.transfor_matrix = transfor_matrix
    
    def transformation_matrix(self):
        width = self.width
        height = int(self.f_dim[1] / self.f_dim[0] * width)
        pad = self.pad
        src = np.float32(self.corners)
        dst = np.float32([[pad, pad],
                          [width + pad, pad],
                          [pad, height + pad],
                          [width + pad, height + pad]])
        self.transfor_matrix = cv.getPerspectiveTransform(src, dst)
        return self.transfor_matrix
    
    def img_perspective_correction(self, frame=None):
        width = self.width
        height = int(self.f_dim[1] / self.f_dim[0] * width)
        pad = self.pad * 2
        if frame is None:
            frame = self.frame.img
        self.frame.img = cv.warpPerspective(frame,
                                            self.transfor_matrix,
                                            tuple(map(int, (width + pad, height + pad)))
                                            )
        return self.frame
    
    def point_perspective_correction(self, points):
        src = np.array([points], dtype=np.float32)
        return cv.perspectiveTransform(src, self.transfor_matrix)[0]
    
    def point_perspective_inv_correction(self, points):
        src = np.array([points], dtype=np.float32)
        return cv.perspectiveTransform(src, np.linalg.inv(self.transfor_matrix))[0]


class Corners:
    def __init__(self, frame=Frame()):
        self.frame = frame
        self.corners = []
    
    def draw(self, frame=None):
        if frame is None:
            frame = self.frame
        for item in self.corners:
            item.draw(frame)
    
    def load_corners(self, parameters):
        color = blue
        for coord in parameters:
            self.corners.append(Point(tuple(coord)))


class Sectors:
    def __init__(self, frame=Frame()):
        self.frame = frame
        self.items = []
    
    def draw(self, frame=None):
        if frame is None:
            frame = self.frame
        for item in self.items:
            item.draw(frame)
    
    def load_sectors(self, parameters):
        color = green
        for item in parameters['sectors']:
            item_type = item[2]
            pt = item[:1][0]
            if item_type == 0 or item_type == 'rectangle':
                self.items.append(Rectangle(
                    pt,
                    item[1],
                    color))
            elif item_type == 1 or item_type == 'circle':
                self.items.append(Circle(
                    pt,
                    item[1],
                    color
                    ))
            elif item_type == 2 or item_type == 'ellipse':
                self.items.append(Ellipse(
                    pt,
                    item[1],
                    color
                    ))

class Items:
    def __init__(self, frame=Frame()):
        self.frame = frame
        self.items = []
    
    def draw(self, frame=None):
        if frame is None:
            frame = self.frame
        for item in self.items:
            item.draw(frame)
    
    def load_items(self, parameters):
        color = red
        for item in parameters['objects']:
            item_type = item[2]
            pt = item[:1][0]
            if item_type == 0 or item_type == 'rectangle':
                self.items.append(Rectangle(
                    pt,
                    item[1],
                    color))
            elif item_type == 1 or item_type == 'circle':
                self.items.append(Circle(
                    pt,
                    item[1],
                    color
                    ))
            elif item_type == 2 or item_type == 'ellipse':
                self.items.append(Ellipse(
                    pt,
                    item[1],
                    color
                    ))


class Item:
    def __init__(self, coord=None, tag=None, color=blue, shape=None, thickness=2, line_type=8):
        self.coord = coord
        self.tag = tag
        self.shape = shape
        self.color = color
        self.thickness = thickness
        self.line_type = line_type


class Point(Item):
    def __init__(self, coord=None, tag=None, color=blue, shape='point',
                 marker_type=0, marker_size=16, thickness=1, line_type=8):
        Item.__init__(self, coord, tag, color, shape, thickness, line_type)
        self.marker_type = marker_type
        self.marker_size = marker_size
    
    def draw(self, frame):
        cv.drawMarker(frame.img, tuple(map(int, self.coord)), self.color, self.marker_type,
                      self.marker_size, self.thickness, self.line_type)


class Rectangle(Item):
    def __init__(self, coord=None, tag=None, color=red, shape='rectangle',
                 thickness=2, line_type=8):
        Item.__init__(self, coord, tag, color, shape, thickness, line_type)
    
    def draw(self, frame):
        w, h = ((self.coord[1][0] - self.coord[0][0]),
                (self.coord[1][1] - self.coord[0][1]))
        cv.rectangle(frame.img,
                     tuple(map(int, (self.coord[0][0] - w,
                                     self.coord[0][1] - h))),
                     self.coord[1],
                     self.color,
                     self.thickness, self.line_type)


class Circle(Item):
    def __init__(self, coord=None, tag=None, color=red, shape='circle',
                 thickness=2, line_type=8):
        Item.__init__(self, coord, tag, color, shape, thickness, line_type)
    
    def draw(self, frame):
        r = abs(self.coord[1][0] - self.coord[0][0])
        cv.circle(frame.img, tuple(map(int, self.coord[0])), int(r), self.color, self.thickness,
                  self.line_type)


class Ellipse(Item):
    def __init__(self, coord=None, tag=None, color=red, shape='ellipse',
                 thickness=2, line_type=8,
                 angle=0, start_angle=0, end_angle=360):
        Item.__init__(self, coord, tag, color, shape, thickness, line_type)
        self.angle = angle
        self.start_angle = start_angle
        self.end_angle = end_angle
    
    def draw(self, frame):
        w, h = (self.coord[1][0] - self.coord[0][0],
                self.coord[1][1] - self.coord[0][1])
        distance_x = abs(self.coord[1][0] - (self.coord[0][0] - w)) // 2
        distance_y = abs(self.coord[1][1] - (self.coord[0][1] - h)) // 2
        cv.ellipse(frame.img, tuple(map(int, self.coord[0])),
                   tuple(map(int, (distance_x, distance_y))),
                   self.angle, self.start_angle, self.end_angle, self.color,
                   self.thickness, self.line_type)


class Tracker:
    def __init__(self, data, parts=None, transform=None):
        self.data = data
        self.parts = parts
        self.transform = transform
        if self.transform is not None:
            self.transform_points()
    
    def transform_points(self):
        for part in self.parts:
            points = self.data[self.data.columns.get_level_values('scorer')[0]][part][['x', 'y']].values
            corrected = self.transform.point_perspective_correction(points)
            self.data[self.data.columns.get_level_values('scorer')[0]][part][['x', 'y']] = corrected
    
    def bodypart_data(self, part):
        return self.data[self.data.columns.get_level_values('scorer')[0]][part][
            ['x', 'y', 'likelihood']]
    
    def bodypart_points(self, part, fnum):
        return self.data[self.data.columns.get_level_values('scorer')[0]][part].loc[fnum]
    
    def selected_bodyparts(self, fnum):
        return self.data[self.data.columns.get_level_values('scorer')[0]][self.parts].loc[fnum]


class Parameters:
    
    def __init__(self):
        self.capture = None
        self.video_file = None
        self.video_frame_count = None
        self.video_fps = None
        self.video_length_ms = None
        self.video_original_size = None
        self.video_corrected_size = None
        self.analysis_start_frame = None
        self.analysis_end_frame = None
        self.experiment = None
        self.experiment_phase = None
        self.field_width = None
        self.field_height = None
        self.field_shape = None
        self.corners = None
        self.objects = None
        self.sectors = None
        self.transformation_matrix = None
    
    def as_dict(self):
        out = dict(
            video_file=str(self.video_file),
            video_frame_count=self.video_frame_count,
            video_fps=self.video_fps,
            video_length_ms=self.video_length_ms,
            video_original_size=self.video_original_size,
            video_corrected_size=self.video_corrected_size,
            analysis_start_frame=self.analysis_start_frame,
            analysis_end_frame=self.analysis_end_frame,
            experiment=self.experiment,
            experiment_phase=self.experiment_phase,
            field_width=self.field_width,
            field_height=self.field_height,
            field_shape=self.field_shape,
            corners=[(x.coord, x.tag, x.shape) for x in self.corners],
            objects=[(x.coord, x.tag, x.shape) for x in self.objects],
            sectors=[(x.coord, x.tag, x.shape) for x in self.sectors],
            transformation_matrix=self.transformation_matrix
            )
        return out
    
    def ask_experiment_settings(self):
        # TODO future: Polygon, ellipse shapes
        
        self.experiment = input('Name or code of the experiment?: ')
        self.experiment_phase = input('Experiment phase of the video?: ')
        
        self.analysis_start_frame = int(input('Frame where analysis should start?: '))
        self.analysis_end_frame = int(input('Frame where analysis should end?: '))
        
        shape = input('Shape of the field (\'0\' or \'rectangle\', \'1\' or \'circle\')?: ')
        if shape in {'rectangle', '0'}:
            self.field_shape = 'rectangle'
            self.field_width = float(input('Width of the rectangle in cm?: '))
            self.field_height = float(input('Height of the rectangle in cm?: '))
        elif shape in {'circle', '1'}:
            self.field_shape = 'circle'
            self.field_width = self.field_height = float(input('Diameter of the circle in cm?: '))
        else:
            print('Shape \'{}\' is not supported!'.format(shape))
        self.corners = [Point(tag='top-left'),
                        Point(tag='top-right'),
                        Point(tag='bottom-left'),
                        Point(tag='bottom-right')]
    
    def ask_objects(self):
        objects = list(map(str.strip, input('Object tags (e.g. \'obj1, obj2b\')?: ').split(',')))
        # TODO: try-catch
        # Items are replaced with correct shape after defining shapes
        self.objects = [None] * len(objects)
        for i, o in enumerate(objects):
            self.objects[i] = Item(None, o)
    
    def set_objects(self):
        # TODO: Make set_objects() universal and use it with set_sectors() too
        def draw_shapes(event, x, y, flags, param):
            if param[5] < len(param[7]):
                if event == cv.EVENT_LBUTTONDOWN:
                    param[1:4] = x, y, True
                elif event == cv.EVENT_MOUSEMOVE and param[3] is True:
                    img1 = param[0].frame.img_org.copy()
                    if flags == cv.EVENT_FLAG_CTRLKEY + cv.EVENT_FLAG_LBUTTON:
                        param[0].frame.img = cv.ellipse(img1,
                                                        (param[1], param[2]),
                                                        (abs(x - param[1]), abs(y - param[2])),
                                                        0, 0, 360, (128, 255, 0), 1)
                        param[6] = 2
                    elif flags == cv.EVENT_FLAG_ALTKEY + cv.EVENT_FLAG_LBUTTON:
                        param[0].frame.img = cv.circle(img1, (param[1], param[2]), abs(x - param[1]), (128, 255, 0))
                        param[6] = 1
                    elif flags == cv.EVENT_FLAG_LBUTTON:
                        param[0].frame.img = cv.rectangle(img1, (param[1], param[2]), (x, y), (128, 255, 0))
                        param[6] = 0
                elif event == cv.EVENT_LBUTTONUP:
                    img_final = param[0].frame.img
                    if param[6] == 2:
                        self.objects[param[5]] = Ellipse(((param[1], param[2]), (x, y)), self.objects[param[5]].tag)
                    elif param[6] == 1:
                        y = (x - param[1]) + param[2]
                        self.objects[param[5]] = Circle(((param[1], param[2]), (x, y)), self.objects[param[5]].tag)
                    elif param[6] == 0:
                        x_pt = param[1] + ((x - param[1]) / 2)
                        y_pt = param[2] + ((y - param[2]) / 2)
                        self.objects[param[5]] = Rectangle(((x_pt, y_pt), (x, y)), self.objects[param[5]].tag)
                    param[0].frame.img_org = img_final
                    self.objects[param[5]].draw(param[0].frame)
                    print(self.objects[param[5]].coord)
                    param[1:4] = 0, 0, False
                    param[5] += 1
                
        o = [x.tag for x in self.objects]
        print(
            'Draw objects: {}. Then hit \'q\' \
            (mouse 1 = rectangle, alt + mouse 1 = circle, ctrl + mouse 1 = ellipse)'.format(o))
        x0, y0, drag, window, current, type = 0, 0, False, 'objects', 0, 0
        cv.namedWindow(window)
        param = [self.capture, x0, y0, drag, window, current, type, self.objects]
        cv.setMouseCallback('objects', draw_shapes, param)
        self.capture.show_img('objects')
    
    def ask_sectors(self):
        sectors = list(map(str.strip, input('Sector tags (e.g. \'center, top-left\')?: ').split(',')))
        # TODO: try-catch
        # Items are replaced with correct shape after defining shapes
        self.sectors = [None] * len(sectors)
        for i, s in enumerate(sectors):
            self.sectors[i] = Item(None, s)
    
    def set_sectors(self):
        def draw_shapes(event, x, y, flags, param):
            if param[5] < len(param[7]):
                if event == cv.EVENT_LBUTTONDOWN:
                    param[1:4] = x, y, True
                elif event == cv.EVENT_MOUSEMOVE and param[3] is True:
                    img1 = param[0].frame.img_org.copy()
                    if flags == cv.EVENT_FLAG_CTRLKEY + cv.EVENT_FLAG_LBUTTON:
                        param[0].frame.img = cv.ellipse(img1,
                                                        (param[1], param[2]),
                                                        (abs(x - param[1]), abs(y - param[2])),
                                                        0, 0, 360, (128, 255, 0), 1)
                        param[6] = 2
                    elif flags == cv.EVENT_FLAG_ALTKEY + cv.EVENT_FLAG_LBUTTON:
                        param[0].frame.img = cv.circle(img1, (param[1], param[2]), abs(x - param[1]), (128, 255, 0))
                        param[6] = 1
                    elif flags == cv.EVENT_FLAG_LBUTTON:
                        param[0].frame.img = cv.rectangle(img1, (param[1], param[2]), (x, y), (128, 255, 0))
                        param[6] = 0
                elif event == cv.EVENT_LBUTTONUP:
                    img_final = param[0].frame.img
                    if param[6] == 2:
                        self.sectors[param[5]] = Ellipse(((param[1], param[2]), (x, y)), self.sectors[param[5]].tag)
                    elif param[6] == 1:
                        self.sectors[param[5]] = Circle(((param[1], param[2]), (x, y)), self.sectors[param[5]].tag)
                    elif param[6] == 0:
                        x_pt = param[1] + ((x - param[1]) / 2)
                        y_pt = param[2] + ((y - param[2]) / 2)
                        self.sectors[param[5]] = Rectangle(((x_pt, y_pt), (x, y)), self.sectors[param[5]].tag)
                    param[0].frame.img_org = img_final
                    self.sectors[param[5]].draw(param[0].frame)
                    print(self.sectors[param[5]].coord)
                    param[1:4] = 0, 0, False
                    param[5] += 1
        
        o = [x.tag for x in self.sectors]
        print(
            'Draw sectors: {}. Then hit \'q\' \
            (mouse 1 = rectangle, alt + mouse 1 = circle, ctrl + mouse 1 = ellipse)'.format(o))
        x0, y0, drag, window, current, type = 0, 0, False, 'sectors', 0, 0
        cv.namedWindow(window)
        param = [self.capture, x0, y0, drag, window, current, type, self.sectors]
        cv.setMouseCallback('sectors', draw_shapes, param)
        self.capture.show_img('sectors')
    
    def ask_corners(self):
        # TODO: drag and drop points for refining shape and place or least values that can be changed manually
        def draw_markers(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONUP:
                
                if param[1] < len(self.corners):
                    self.corners[param[1]].coord = (x, y)
                    self.corners[param[1]].color = (0, 0, 255)
                    print(self.corners[param[1]].coord)
                    self.corners[param[1]].draw(self.capture.frame)
                    param[1] += 1
        
        if self.field_shape == 'rectangle' or self.field_shape == 'circle':
            print('Select corners: top left, top right, bottom left, bottom right. Then hit \'q\'.')
            cv.namedWindow('corners')
            window, current = 'corners', 0
            param = [window, current]
            cv.setMouseCallback('corners', draw_markers, param)
            self.capture.show_img('corners')
        
        # TODO: better circle shape correction
        # elif self.field_shape == 'circle':
        #     print('Draw circle field shape. Then hit \'q\'.')
        #     cv.namedWindow('corners')
        #     # window, current = 'corners', 0
        #     # param = [window, current]
        #     # cv.setMouseCallback('corners', draw_markers, param)
        #     x0, y0, drag, window, done = 0, 0, False, 'corners', False
        #     param = [self.capture, x0, y0, drag, window, done]
        #     cv.setMouseCallback('corners', draw_field, param)
        #     self.capture.show_img('corners')
    
    def perspective_correction(self):
        w, h = (self.corners[3].coord[0] - self.corners[0].coord[0],
                self.corners[3].coord[1] - self.corners[0].coord[1])
        
        h_cor = int(self.field_height / self.field_width * w)
        
        coords = [x.coord for x in self.corners]
        src = np.float32(coords)
        
        # +100 Adds padding to perspective correction
        pad = 100
        dst = np.float32([[pad, pad], [w + pad, pad],
                          [pad, h_cor + pad], [w + pad, h_cor + pad]])
        
        
        # Apply perspective correction to image
        
        self.transformation_matrix = cv.getPerspectiveTransform(src, dst)
        self.capture.frame.img = self.capture.frame.img_org = cv.warpPerspective(self.capture.frame.img.copy(),
                                                                             self.transformation_matrix,
                                                                             (w + (pad * 2), h_cor + (pad * 2)))
        self.video_corrected_size = self.capture.frame.size()
        self.capture.show_img('corners')
