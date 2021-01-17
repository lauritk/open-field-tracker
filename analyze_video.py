import time
import sys
import utils
import video
import cv2 as cv
import numpy as np
from pathlib import Path
import geometry

if __name__ == "__main__":
    start_time = time.time()
    print("Analysis started!")
    
    input_file = Path(sys.argv[1])
    config_file = input_file.parent / (input_file.stem + "_config.ini")
    project_file = input_file.parent / (input_file.stem + "_project_config.ini")

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
    
    output_file = input_file.parent / (input_file.stem + "_results.txt")
    
    parameters = utils.load_parameters(config_file)
    project = utils.load_parameters(project_file)

    deep_file = input_file.parent / (input_file.stem + project['deeplabcut_name'] + ".h5")
    
    # Maybe not needed. Code is so fast now that it can do everything :)
    start = parameters['analysis_start_frame']
    end = parameters['analysis_end_frame']

    # cm
    distance_th1 = project['distance_th1']
    angle_th1 = project['angle_th']
    # cm
    distance_th2 = project['distance_th2']

    # TODO: Fix for not needing the video
    cap = video.Capture(str(input_file))
    cap.capture_frame()
    cap.capture.set(cv.CAP_PROP_POS_FRAMES, start)

    # Transform original corner coordinates to corrected frame dimensions
    trans = video.Transformation(cap.frame,
                                 corners=parameters['corners'],
                                 transfor_matrix=parameters['transformation_matrix'])
    corners = video.Corners(cap.frame)
    corners.load_corners(trans.point_perspective_correction(trans.corners))
    
    items = video.Items(cap.frame)
    items.load_items(parameters)
    sectors = video.Sectors(cap.frame)
    sectors.load_sectors(parameters)

    save_parts = project['save_parts']
    angle_parts = project['angle_parts']
    sector_parts = project['sector_parts']
    sectors_criteria = project['sectors_criteria']
    track_part = project['track_part']
    base_part = project['base_part']
    explore_part = project['explore_part']

    tracker_data = video.Tracker(utils.load_deeplab_h5(deep_file), save_parts, trans)
    likelihood_limit = project['likelihood_limit']
    still_limit = project['still_limit']
    visit_gap = project['visit_gap']
    visit_length = project['visit_length']
    
    n = tracker_data.data.shape[0]
    n_parts = len(save_parts)
    n_items = len(items.items)
    n_sectors = len(sectors.items)
    
    items_intersection_pt = np.zeros([n_parts, n_items, n, 2], dtype=np.float64)
    items_intersection_pt_cm = np.zeros([n_parts, n_items, n, 2], dtype=np.float64)
    items_centroid_distance = np.zeros([n_parts, n_items, n], dtype=np.float64)
    items_edge_distance = np.zeros([n_parts, n_items, n], dtype=np.float64)
    items_point_inside = np.zeros([n_parts, n_items, n], dtype=np.float64)
    items_explore = np.zeros([n_items, n], dtype=np.float64)

    sectors_intersection_pt = np.zeros([n_parts, n_sectors, n, 2], dtype=np.float64)
    sectors_intersection_pt_cm = np.zeros([n_parts, n_sectors, n, 2], dtype=np.float64)
    sectors_centroid_distance = np.zeros([n_parts, n_sectors, n], dtype=np.float64)
    sectors_edge_distance = np.zeros([n_parts, n_sectors, n], dtype=np.float64)
    sectors_point_inside = np.zeros([n_parts, n_sectors, n], dtype=np.float64)
    sectors_inside = np.zeros([n_sectors, n], dtype=np.float64)
    
    # Collects bodypart x, y, likelihood (order of 'save_parts' variable)
    bodyparts = np.zeros([n_parts, n, 3], dtype=np.float64)
    bodyparts_cm = np.zeros([n_parts, n, 3], dtype=np.float64)
    
    bodypart_item_angle = np.zeros([n_items, n], dtype=np.float64)
    bodypart_item_angle_norm = np.zeros([n_items, n], dtype=np.float64)

    bodypart_sector_angle = np.zeros([n_sectors, n], dtype=np.float64)
    bodypart_sector_angle_norm = np.zeros([n_sectors, n], dtype=np.float64)
    
    animal_distance = np.zeros(n - 1, dtype=np.float64)
    animal_speed = np.zeros(n - 1, dtype=np.float64)

    for i, part in enumerate(save_parts):
        
        # Interpolates coordinates below 'likelihood_limit'
        bodyparts[i, :] = tracker_data.bodypart_data(part).values
        bodyparts[i, :, :2] = utils.correct_coordinates(bodyparts[i, :, :2], bodyparts[i, :, 2], likelihood_limit)
        bodyparts_cm[i, :] = utils.data_px_to_cm(bodyparts[i, :], corners, parameters)

        for ix, item in enumerate(items.items):
            
            item_pt = np.array(item.coord[0])
            item_coords = np.repeat([item_pt], n, axis=0)
            item_coords_cm = utils.data_px_to_cm(item_coords, corners, parameters)
            
            item_w = item.coord[1][0] - item.coord[0][0]
            item_h = item.coord[1][1] - item.coord[0][1]
            
            if item.shape == 'circle':
                items_intersection_pt[i, ix, :, :] = geometry.line_segment_circle_intersection(
                    item_coords, item_w, item_coords, bodyparts[i, :, :2], 'start'
                    )
            elif item.shape == ' ellipse':
                items_intersection_pt[i, ix, :, :] = geometry.line_ellipse_intersection(
                    item_coords, item_w, item_h, item_coords, bodyparts[i, :, :2], 'start'
                    )
            else:
                items_intersection_pt[i, ix, :, :] = geometry.line_rectangle_intersection(
                    item_coords, item_w, item_h, item_coords, bodyparts[i, :, :2], 'start'
                    )
            transposed_bp = bodyparts[i, :, :2].T
            transposed_bp_cm = bodyparts_cm[i, :, :2].T
            
            items_intersection_pt_cm[i, ix, :, :] = utils.data_px_to_cm(items_intersection_pt[i, ix, :, :], corners, parameters)
            items_centroid_distance[i, ix, :] = utils.sqrt_sum(item_coords_cm.T, transposed_bp_cm)
            items_edge_distance[i, ix, :] = utils.sqrt_sum(items_intersection_pt_cm[i, ix].T, transposed_bp_cm)
            items_centroid_to_edge = utils.sqrt_sum(items_intersection_pt_cm[i, ix].T, item_coords_cm.T)
            items_point_inside[i, ix, :] = (items_centroid_to_edge > items_centroid_distance[i, ix, :])
            
        for ix, sector in enumerate(sectors.items):
            item_pt = np.array(sector.coord[0])
            item_coords = np.repeat([item_pt], n, axis=0)
            item_coords_cm = utils.data_px_to_cm(item_coords, corners, parameters)
    
            item_w = sector.coord[1][0] - sector.coord[0][0]
            item_h = sector.coord[1][1] - sector.coord[0][1]

            if sector.shape == 'circle':
                sectors_intersection_pt[i, ix, :, :] = geometry.line_segment_circle_intersection(
                    item_coords, item_w, item_coords, bodyparts[i, :, :2], 'start'
                    )
            elif sector.shape == ' ellipse':
                sectors_intersection_pt[i, ix, :, :] = geometry.line_ellipse_intersection(
                    item_coords, item_w, item_h, item_coords, bodyparts[i, :, :2], 'start'
                    )
            else:
                sectors_intersection_pt[i, ix, :, :] = geometry.line_rectangle_intersection(
                    item_coords, item_w, item_h, item_coords, bodyparts[i, :, :2], 'start'
                    )
            transposed_bp = bodyparts[i, :, :2].T
            transposed_bp_cm = bodyparts_cm[i, :, :2].T

            sectors_intersection_pt_cm[i, ix, :, :] = utils.data_px_to_cm(sectors_intersection_pt[i, ix, :, :], corners, parameters)
            sectors_centroid_distance[i, ix, :] = utils.sqrt_sum(item_coords_cm.T, transposed_bp_cm)
            sectors_edge_distance[i, ix, :] = utils.sqrt_sum(sectors_intersection_pt_cm[i, ix].T, transposed_bp_cm)
            sectors_centroid_to_edge = utils.sqrt_sum(sectors_intersection_pt_cm[i, ix].T, item_coords_cm.T)
            sectors_point_inside[i, ix, :] = (sectors_centroid_to_edge > sectors_centroid_distance[i, ix, :])

    for ix, item in enumerate(items.items):
        item_pt = np.array(item.coord[0])
        item_coords = np.repeat([item_pt], n, axis=0)

        
        bp1_idx = save_parts.index(angle_parts[0])
        bp2_idx = save_parts.index(angle_parts[1])
        bp1 = bodyparts[bp1_idx, :, :2]
        bp2 = bodyparts[bp2_idx, :, :2]

        bp1c = bodyparts_cm[bp1_idx, :, :2]
        bp2c = bodyparts_cm[bp2_idx, :, :2]
        item_coords_cm = utils.data_px_to_cm(item_coords, corners, parameters)
        
        # Angle from centroid of the item
        bodypart_item_angle[ix] = utils.angle2(bp1c, item_coords_cm, bp2c).squeeze()
        bodypart_item_angle_norm[ix] = np.min((360 - bodypart_item_angle[ix, :],
                                                    bodypart_item_angle[ix, :]),
                                                   axis=0)

    for ix, sector in enumerate(sectors.items):
        item_pt = np.array(sector.coord[0])
        item_coords = np.repeat([item_pt], n, axis=0)
    
        bp1_idx = save_parts.index(angle_parts[0])
        bp2_idx = save_parts.index(angle_parts[1])
        bp1 = bodyparts[bp1_idx, :, :2]
        bp2 = bodyparts[bp2_idx, :, :2]

        bp1c = bodyparts_cm[bp1_idx, :, :2]
        bp2c = bodyparts_cm[bp2_idx, :, :2]
        item_coords_cm = utils.data_px_to_cm(item_coords, corners, parameters)
    
        # # Angle from centroid of the item
        bodypart_sector_angle[ix] = utils.angle2(bp1c, item_coords_cm, bp2c).squeeze()
        bodypart_sector_angle_norm[ix] = np.min((360 - bodypart_item_angle[ix, :],
                                               bodypart_item_angle[ix, :]),
                                              axis=0)
        
    for i, item in enumerate(items.items):
        part = save_parts.index(explore_part)
        dist = items_edge_distance[part, i, :].T < distance_th1
        dist2 = items_edge_distance[part, i, :].T < distance_th2
        angle = bodypart_item_angle_norm[i, :].T > angle_th1
        dist_angle = np.logical_and(dist, angle)
        dist_or = np.logical_or(dist2, items_point_inside[part, i, :].T)
        items_explore[i, :] = np.logical_or(dist_or, dist_angle)

    for i, sector in enumerate(sectors.items):
        part = [save_parts.index(x) for x in sector_parts]
        sector_sum = np.sum(sectors_point_inside[part, i, :], axis=0)
        sectors_inside[i, :] = sector_sum >= sectors_criteria
            
    # Distance and speed from base_part (body_center)
    data = bodyparts[save_parts.index(base_part)][:, :2][start:end]

    f_count = end - start
    frame_time = cap.duration_in_s() / cap.total_frames
    duration = f_count * frame_time
    
    corrected = utils.data_px_to_cm(data, corners, parameters)
    animal_distance = utils.moved_distance(corrected)
    still = np.where(animal_distance < still_limit)[0]
    not_still = np.where(animal_distance >= still_limit)[0]
    sum_dist = utils.sum_distance_cm(animal_distance)
    animal_speed = utils.speed_cm_s(animal_distance, duration, f_count)
    animal_speed_mean = np.mean(animal_speed)
    corrected_speed_mean = np.mean(animal_speed[not_still])
    
    np.save(items_intersection_pt_file, items_intersection_pt)
    np.save(items_intersection_pt_cm_file, items_intersection_pt_cm)
    np.save(items_centroid_distance_file, items_centroid_distance)
    np.save(items_edge_distance_file, items_edge_distance)
    np.save(items_point_inside_file, items_point_inside)
    np.save(items_explore_file, items_explore)
    np.save(sectors_intersection_pt_file, sectors_intersection_pt)
    np.save(sectors_intersection_pt_cm_file, sectors_intersection_pt_cm)
    np.save(sectors_centroid_distance_file, sectors_centroid_distance)
    np.save(sectors_edge_distance_file, sectors_edge_distance)
    np.save(sectors_point_inside_file, sectors_point_inside)
    np.save(sectors_inside_file, sectors_inside)
    np.save(bodyparts_file, bodyparts)
    np.save(bodyparts_cm_file, bodyparts_cm)
    np.save(bodypart_item_angle_file, bodypart_item_angle)
    np.save(bodypart_item_angle_norm_file, bodypart_item_angle_norm)
    np.save(bodypart_sector_angle_file, bodypart_sector_angle)
    np.save(bodypart_sector_angle_norm_file, bodypart_sector_angle_norm)
    np.save(animal_distance_file, animal_distance)
    np.save(animal_speed_file, animal_speed)

    items_explored = [None] * n_items
    items_visit_count = [None] * n_items
    items_visit_idx = [None] * n_items
    for idx, x in enumerate(items_explore[:, start:end]):
        items_explored[idx] = (items.items[idx].tag, np.sum(x))
        items_visit_idx[idx] = utils.visits_idx(x, visit_gap, visit_length)
        items_visit_count[idx] = [len(x) for x in items_visit_idx[idx]][0]

    sectors_in = [None] * n_sectors
    sectors_visit_count = [None] * n_sectors
    sectors_visit_idx = [None] * n_sectors
    for idx, x in enumerate(sectors_inside[:, start:end]):
        sectors_in[idx] = (sectors.items[idx].tag, np.sum(x))
        sectors_visit_idx[idx] = utils.visits_idx(x, visit_gap, visit_length)
        sectors_visit_count[idx] = [len(x) for x in sectors_visit_idx[idx]][0]

    # https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
    def pretty(d, indent=0):
        out = ""
        for key, value in d.items():
            out = out + '\n' + ('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent + 1)
            elif isinstance(value, list):
                if len(value) > 1 and isinstance(value[0], tuple):
                    for x in value:
                        out = out + '\n' + ('\t' * (indent + 1) + str(x))
                else:
                    out = out + '\n' + ('\t' * (indent + 1) + str(value[0]))
            else:
                out = out + '\n' + ('\t' * (indent + 1) + str(value))
        return out
    
    out = {
        'total_frames': f_count,
        'duration_s': duration,
        'items_explored': items_explored,
        'items_visits': items_visit_count,
        'items_visit_idx': items_visit_idx,
        'sectors_inside': sectors_in,
        'sectors_visits': sectors_visit_count,
        'sectors_visit_idx': sectors_visit_idx,
        'distance_cm': np.sum(animal_distance),
        'frames_still': len(still),
        'raw_speed_cm_s': animal_speed_mean,
        'corrected_speed_cm_s': corrected_speed_mean
        }

    print(pretty(out))
    f = open(output_file, "w")
    f.write(pretty(out))
    f.close()
    end_time = time.time()
    print('Analysis ended!')
    print('total time (s)= ' + str(end_time - start_time))