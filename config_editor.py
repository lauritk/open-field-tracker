import time
import sys
import utils
import video
import math
import cv2 as cv
from pathlib import Path

input_file = Path(sys.argv[1])
config_file = input_file.parent / (input_file.stem + "_config.ini")
project_file = input_file.parent / (input_file.stem + "_project_config.ini")

parameters = utils.load_parameters(config_file)
project = utils.load_parameters(project_file)

start = parameters['analysis_start_frame']
end = parameters['analysis_end_frame']
cap = video.Capture(str(input_file))
cap.capture_frame()
cap.capture.set(cv.CAP_PROP_POS_FRAMES, start)

trans = video.Transformation(cap.frame,
                             corners=parameters['corners'],
                             transfor_matrix=parameters['transformation_matrix'],
                             f_dim=(parameters['field_width'], parameters['field_height']))
trans.img_perspective_correction()

items = video.Items(cap.frame)
items.load_items(parameters)

sectors = video.Sectors(cap.frame)
sectors.load_sectors(parameters)

corners = video.Corners(cap.frame)
corners.load_corners(trans.point_perspective_correction(trans.corners))

def nothing(val):
    # OpenCV trackbar trick
    pass

print("Edit coordinates and press 'k' to continue...")
# Save corners after inverse transform
cv.namedWindow('corners')
cv.createTrackbar('X', "corners", 0, cap.frame.size()[0], nothing)
cv.createTrackbar('Y', "corners", 0, cap.frame.size()[1], nothing)
org = cap.frame.img
new = [video.Point(tag='top-left'),
       video.Point(tag='top-right'),
       video.Point(tag='bottom-left'),
       video.Point(tag='bottom-right')]
for idx, corner in enumerate(corners.corners):
    cv.setTrackbarPos('X', 'corners', int(corner.coord[0]))
    cv.setTrackbarPos('Y', 'corners', int(corner.coord[1]))
    wait = True
    while(wait):
        coord = (cv.getTrackbarPos('X', 'corners'), cv.getTrackbarPos('Y', 'corners'))
        frame = cv.drawMarker(org.copy(), coord, (255, 255, 0), 0, 16, 1, 8)
        if cv.waitKey(1) & 0xFF == ord('k'):
            new[idx].coord = tuple(map(math.ceil, *trans.point_perspective_inv_correction([coord])))
            # print(corner.coord)
            org = frame.copy()
            wait = False
        cap.show('corners', frame)
cv.destroyAllWindows()


cor = [(x.coord, x.tag, x.shape) for x in new]
cap.capture_frame()
trans.frame = cap.frame
trans = video.Transformation(cap.frame,
                             corners=cor,
                             f_dim=(parameters['field_width'], parameters['field_height']))
# trans.corners = [x.coord for x in new]
trans.transformation_matrix()
trans.img_perspective_correction(cap.frame.img_org.copy())
org = cap.frame.img.copy()
parameters['corners'] = cor
parameters['transformation_matrix'] = trans.transfor_matrix

cv.namedWindow('items')
cv.createTrackbar('X', "items", 0, cap.frame.size()[0], nothing)
cv.createTrackbar('Y', "items", 0, cap.frame.size()[1], nothing)
cv.createTrackbar('W', "items", 0, cap.frame.size()[0], nothing)
cv.createTrackbar('H', "items", 0, cap.frame.size()[1], nothing)
for idx, item in enumerate(items.items):
    cv.setTrackbarPos('X', 'items', int(item.coord[0][0]))
    cv.setTrackbarPos('Y', 'items', int(item.coord[0][1]))
    cv.setTrackbarPos('W', 'items', int(item.coord[1][0]) - int(item.coord[0][0]))
    cv.setTrackbarPos('H', 'items', int(item.coord[1][1]) - int(item.coord[0][1]))
    wait = True
    while(wait):
        if item.shape == 'circle':
            frame = cv.circle(org.copy(), (cv.getTrackbarPos('X', 'items'), cv.getTrackbarPos('Y', 'items')),
                              cv.getTrackbarPos('W', 'items'), (255, 255, 0), 1, 8)
        elif item.shape == 'ellipse':
            frame = cv.ellipse(org.copy(), (cv.getTrackbarPos('X', 'items'), cv.getTrackbarPos('Y', 'items')),
                               (cv.getTrackbarPos('W', 'items'), cv.getTrackbarPos('H', 'items')), 0, 0, 360, (255, 255, 0), 1, 8)
        else:
            frame = cv.rectangle(org.copy(), (cv.getTrackbarPos('X', 'items') - cv.getTrackbarPos('W', 'items'), cv.getTrackbarPos('Y', 'items') - cv.getTrackbarPos('H', 'items')),
                              (cv.getTrackbarPos('X', 'items') + cv.getTrackbarPos('W', 'items'), cv.getTrackbarPos('Y', 'items')+ cv.getTrackbarPos('H', 'items')), (255, 255, 0), 1, 8)
        if cv.waitKey(1) & 0xFF == ord('k'):
            
            item.coord = tuple(map(int, (cv.getTrackbarPos('X', 'items'), cv.getTrackbarPos('Y', 'items')))), \
                         tuple(map(int, (cv.getTrackbarPos('X', 'items') + cv.getTrackbarPos('W', 'items'), cv.getTrackbarPos('Y', 'items') + cv.getTrackbarPos('W', 'items'))))
            org = frame.copy()
            wait = False
        cap.show('items', frame)

cv.destroyAllWindows()
parameters['objects'] = [(x.coord, x.tag, x.shape) for x in items.items]

cv.namedWindow('items')
cv.createTrackbar('X', "items", 0, cap.frame.size()[0], nothing)
cv.createTrackbar('Y', "items", 0, cap.frame.size()[1], nothing)
cv.createTrackbar('W', "items", 0, cap.frame.size()[0], nothing)
cv.createTrackbar('H', "items", 0, cap.frame.size()[1], nothing)
for idx, item in enumerate(sectors.items):
    cv.setTrackbarPos('X', 'items', int(item.coord[0][0]))
    cv.setTrackbarPos('Y', 'items', int(item.coord[0][1]))
    cv.setTrackbarPos('W', 'items', int(item.coord[1][0]) - int(item.coord[0][0]))
    cv.setTrackbarPos('H', 'items', int(item.coord[1][1]) - int(item.coord[0][1]))
    wait = True
    while (wait):
        if item.shape == 'circle':
            frame = cv.circle(org.copy(), (cv.getTrackbarPos('X', 'items'), cv.getTrackbarPos('Y', 'items')),
                              cv.getTrackbarPos('W', 'items'), (255, 255, 0), 1, 8)
        elif item.shape == 'ellipse':
            frame = cv.ellipse(org.copy(), (cv.getTrackbarPos('X', 'items'), cv.getTrackbarPos('Y', 'items')),
                               (cv.getTrackbarPos('W', 'items'), cv.getTrackbarPos('H', 'items')), 0, 0, 360,
                               (255, 255, 0), 1, 8)
        else:
            frame = cv.rectangle(org.copy(), (cv.getTrackbarPos('X', 'items') - cv.getTrackbarPos('W', 'items'),
                                              cv.getTrackbarPos('Y', 'items') - cv.getTrackbarPos('H', 'items')),
                                 (cv.getTrackbarPos('X', 'items') + cv.getTrackbarPos('W', 'items'),
                                  cv.getTrackbarPos('Y', 'items') + cv.getTrackbarPos('H', 'items')), (255, 255, 0), 1,
                                 8)
        if cv.waitKey(1) & 0xFF == ord('k'):
            item.coord = tuple(map(int, (cv.getTrackbarPos('X', 'items'), cv.getTrackbarPos('Y', 'items')))), \
                         tuple(map(int, (cv.getTrackbarPos('X', 'items') + cv.getTrackbarPos('W', 'items'), cv.getTrackbarPos('Y', 'items') + cv.getTrackbarPos('W', 'items'))))
            org = frame.copy()
            wait = False
        cap.show('items', frame)

cv.destroyAllWindows()
parameters['sectors'] = [(x.coord, x.tag, x.shape) for x in sectors.items]
utils.save_parameters(config_file, parameters)

