import video
import utils
import sys
import time
import cv2 as cv
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    
    start_time = time.time()
    print("Video plotting started!")
    
    green = (70, 121, 0)
    pink = (255, 159, 228)
    yellow = (0, 123, 237)
    blue = (222, 154, 2)
    red = (51, 0, 155)
    
    def blend_mask(img, mask):
        img2 = np.zeros(img.shape, dtype=np.uint8)
        img2[:, :, :] = blue
        img_bg = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
        img_fg = cv.bitwise_and(img2, img2, mask=mask)
        blend = cv.add(img_bg, img_fg)
        out = cv.addWeighted(img, 0.4, blend, 0.6, 0)
        return out
    
    input_file = Path(sys.argv[1])
    config_file = input_file.parent / (input_file.stem + "_config.ini")
    project_file = input_file.parent / (input_file.stem + "_project_config.ini")
    video_plot_file = input_file.parent / (input_file.stem + "_plot.mp4")

    # TODO: combine some of the arrays
    items_intersection_pt_file = input_file.parent / (input_file.stem + "_intersection.npy")
    items_intersection_pt_cm_file = input_file.parent / (input_file.stem + "_intersection_cm.npy")
    items_centroid_distance_file = input_file.parent / (input_file.stem + "_centroid_distance.npy")
    items_edge_distance_file = input_file.parent / (input_file.stem + "_edge_distance.npy")
    items_point_inside_file = input_file.parent / (input_file.stem + "_items_point_inside.npy")
    items_explore_file = input_file.parent / (input_file.stem + "_items_explore.npy")
    sectors_intersection_pt_cm_file = input_file.parent / (input_file.stem + "_intersection_sector_cm.npy")
    sectors_intersection_pt_file = input_file.parent / (input_file.stem + "_intersection_sector.npy")
    sectors_centroid_distance_file = input_file.parent / (input_file.stem + "_centroid_distance_sector.npy")
    sectors_edge_distance_file = input_file.parent / (input_file.stem + "_edge_distance_sector.npy")
    sectors_point_inside_file = input_file.parent / (input_file.stem + "_sectors_point_inside.npy")
    sectors_inside_file = input_file.parent / (input_file.stem + "_sectors_inside.npy")
    bodyparts_file = input_file.parent / (input_file.stem + "_bodyparts.npy")
    bodyparts_cm_file = input_file.parent / (input_file.stem + "_bodyparts_cm.npy")
    bodypart_item_angle_file = input_file.parent / (input_file.stem + "_angle.npy")
    bodypart_item_angle_norm_file = input_file.parent / (input_file.stem + "_norm_angle.npy")
    bodypart_sector_angle_file = input_file.parent / (input_file.stem + "_angle_sector.npy")
    bodypart_sector_angle_norm_file = input_file.parent / (input_file.stem + "_norm_angle_sector.npy")
    animal_distance_file = input_file.parent / (input_file.stem + "_animal_distance.npy")
    animal_speed_file = input_file.parent / (input_file.stem + "_animal_speed.npy")
    
    parameters = utils.load_parameters(config_file)
    project = utils.load_parameters(project_file)

    deep_file = input_file.parent / (input_file.stem + project['deeplabcut_name'] + ".h5")

    prev = False
    if len(sys.argv) > 2:
        if sys.argv[2] == '--preview':
            prev = True
    
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
    
    
    items_intersection_pt = np.load(items_intersection_pt_file)
    items_centroid_distance = np.load(items_centroid_distance_file)
    items_edge_distance = np.load(items_edge_distance_file)
    items_point_inside = np.load(items_point_inside_file)
    items_explore = np.load(items_explore_file)
    sectors_intersection_pt = np.load(sectors_intersection_pt_file)
    sectors_centroid_distance = np.load(sectors_centroid_distance_file)
    sectors_edge_distance = np.load(sectors_edge_distance_file)
    sectors_point_inside = np.load(sectors_point_inside_file)
    sectors_inside = np.load(sectors_inside_file)
    bodyparts = np.load(bodyparts_file)
    bodypart_item_angle = np.load(bodypart_item_angle_file)
    bodypart_item_angle_norm = np.load(bodypart_item_angle_norm_file)
    bodypart_sector_angle = np.load(bodypart_sector_angle_file)
    bodypart_sector_angle_norm = np.load(bodypart_sector_angle_norm_file)
    animal_distance = np.load(animal_distance_file)
    animal_speed = np.load(animal_speed_file)
    
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    if not prev:
        video_out = cv.VideoWriter(str(video_plot_file), fourcc, cap.fps, cap.frame.size())

    save_parts = project['save_parts']
    angle_parts = project['angle_parts']
    sector_parts = project['sector_parts']
    sectors_criteria = project['sectors_criteria']
    track_part = project['track_part']
    base_part = project['base_part']
    explore_part = project['explore_part']
    
    width = corners.corners[1].coord[0] - corners.corners[0].coord[0]
    height = corners.corners[2].coord[1] - corners.corners[0].coord[1]
    pixels_in_cm_w = width / parameters['field_width']
    pixels_in_cm_h = height / parameters['field_height']

    # cm
    distance_th1 = project['distance_th1']
    angle_th1 = project['angle_th']
    # cm
    distance_th2 = project['distance_th2']
    
    n_parts = len(save_parts)
    n_items = len(items.items)
    n = parameters['video_frame_count']
    
    mask = None
    fn = start
    while cap.is_open() and (fn < end):
        cap.capture_frame()
        trans.img_perspective_correction()
        
        if mask is None:
            mask = np.zeros(cap.frame.size(), dtype=np.uint8)
    
        mask = cv.drawMarker(mask,
                             tuple(map(int, bodyparts[save_parts.index(track_part), cap.frame.num][:2])),
                             (255, 255, 255), cv.MARKER_DIAMOND, 1, 2)
    
        cap.frame.img = blend_mask(cap.frame.img, mask)
    
        items.draw()
        sectors.draw()
        corners.draw()
        
        for i, inside in enumerate(sectors_inside):
            if inside[cap.frame.num]:
                color = red
                sectors.items[i].color = blue
            else:
                color = green
                sectors.items[i].color = green
    
        for i, explore_item in enumerate(items_explore):
            if explore_item[cap.frame.num]:
                color = blue
                items.items[i].color = blue
            else:
                color = yellow
                items.items[i].color = yellow
            intersect_pt = items_intersection_pt[save_parts.index(track_part), i]
    
        cv.line(cap.frame.img,
                tuple(map(int, bodyparts[save_parts.index(angle_parts[0]), cap.frame.num][:2])),
                tuple(map(int, bodyparts[save_parts.index(angle_parts[1]), cap.frame.num][:2])),
                red, 2)

        if prev:
            cap.show()
        else:
            video_out.write(cap.frame.img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        fn += 1
    
    if not prev:
        video_out.release()
    cap.release_capture()
    cv.destroyAllWindows()

    end_time = time.time()
    print('Plotting ended!')
    print('total time (s)= ' + str(end_time - start_time))