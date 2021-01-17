import numpy as np


def rdot(a, b):
    """Element-wise dot product"""
    if a.ndim < 2 or b.ndim < 2:
        return np.dot(a, b)
    return np.einsum('ij,ij->i', a, b)

def line_segment_circle_intersection(c_pt, r, l_start, l_end, closest=None):
    # direction of line
    l = np.subtract(l_end, l_start)
    # line from line start to center of circle
    o_c = np.subtract(l_start, c_pt)
    
    # Quadratic equation
    a = rdot(l, l)
    a2 = 2 * a
    b = 2 * rdot(l, o_c)
    c = rdot(o_c, o_c) - np.square(r)
    ac = 4 * a * c
    c2 = np.square(b) - ac
    with np.errstate(invalid='ignore'):
        # if root root < 0, no intersection.
        root = np.sqrt(c2)
    d1 = (-b - root) / a2
    d2 = (-b + root) / a2
    
    # Fix for multiplying right d with right coordinate
    d1 = d1[:, np.newaxis]
    d2 = d2[:, np.newaxis]
    
    if 'end' == closest:
        closest = np.where(np.abs(d1) > np.abs(d2), d1, d2)
        return l_start + closest * l
    elif 'start' == closest:
        closest = np.where(np.abs(d1) < np.abs(d2), d1, d2)
        return l_start + closest * l
    else:
        return l_start + d1 * l, l_start + d2 * l


def ray_line_intersection(r, l):
    # TODO: https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
    pass

def line_line_intersection(l1, l2):
    # http://paulbourke.net/geometry/pointlineplane/
    # l1 = np.asarray(l1)
    # l2 = np.asarray(l2)
    denom = (l2[1][1] - l2[0][1]) * (l1[1][0] - l1[0][0]) - \
                  (l2[1][0] - l2[0][0]) * (l1[1][1] - l1[0][1])
    numa = (l2[1][0] - l2[0][0]) * (l1[0][1] - l2[0][1]) - \
           (l2[1][1] - l2[0][1]) * (l1[0][0] - l2[0][0])
    numb = (l1[1][0] - l1[0][0]) * (l1[0][1] - l2[0][1]) - \
           (l1[1][1] - l2[0][1]) * (l1[0][0] - l2[0][0])
    
    if denom == 0:
        x = None
        y = None
    else:
        ua = numa / denom
        ub = numb / denom
        
        x = l1[0][0] + ua * (l1[1][0] - l1[0][0])
        y = l1[0][1] + ua * (l1[1][1] - l1[0][1])
    
    return x, y


def line_line_intersection2(l1, l2, point=None):
    # http://paulbourke.net/geometry/pointlineplane/
    # Alternative with cross product https://stackoverflow.com/a/42727584/10111563
    denom = (l2[1, :, 1] - l2[0, :, 1]) * (l1[1, :, 0] - l1[0, :, 0]) - \
            (l2[1, :, 0] - l2[0, :, 0]) * (l1[1, :, 1] - l1[0, :, 1])
    numa = (l2[1, :, 0] - l2[0, :, 0]) * (l1[0, :, 1] - l2[0, :, 1]) - \
           (l2[1, :, 1] - l2[0, :, 1]) * (l1[0, :, 0] - l2[0, :, 0])
    numb = (l1[1, :, 0] - l1[0, :, 0]) * (l1[0, :, 1] - l2[0, :, 1]) - \
           (l1[1, :, 1] - l1[0, :, 1]) * (l1[0, :, 0] - l2[0, :, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        ua = np.divide(numa, denom)
        ub = np.divide(numb, denom)
        # 0 ≤ t and 0 ≤ u ≤ 1.
        if point == 'start':
            check = ((ua <= 0) | (ua >= 1) | (ub <= 0) | (ub >= 1))
            check2 = ((ua >= 0) | (ub <= 0) | (ua >= 1) | (ub >= 1))
 
            # TODO: Fix this. This works now only for rectangles
            check = check.reshape((-1, 4))
            check2 = check2.reshape((-1, 4))
            final = np.any(~check, axis=1)
            check[~final] = check2[~final]
            check = check.reshape((-1))
            
        else:
            check = ((ua <= 0) | (ua >= 1) | (ub <= 0) | (ub >= 1))
        # Would be better to calculate coordinates only for existing intersections
        x = l1[0, :, 0] + ua * (l1[1, :, 0] - l1[0, :, 0])
        y = l1[0, :, 1] + ua * (l1[1, :, 1] - l1[0, :, 1])
        
        x[np.isneginf(x)] = x[check] = np.nan
        y[np.isneginf(y)] = y[check] = np.nan
    
    return np.asarray((x, y)).T

# TODO: Make this work
def line_polygon_intersection(l, p, point=None):
    p_lines = np.array([
        p,
        np.roll(p, 1, axis=0)])
    lines = np.tile(l, (1, len(p), 1))
    out = line_line_intersection2(lines, p_lines, point)
    return out[~np.isnan(out)].reshape((-1, 2))


# TODO: Make this work
def multi_line_polygon_intersection(l, p):
    out = np.zeros((len(l), 2))
    for i, lines in enumerate(l):
        test = line_polygon_intersection(lines, p)
        if len(test) < 2:
            out[i, :] = None
        else:
            out[i, :] = line_polygon_intersection(lines, p)
    return out

def line_rectangle_intersection(center, w, h, l_start, l_end, closest=None):
    polygon = np.array([[center[0, 0] - w, center[0, 1] - h],
                        [center[0, 0] + w, center[0, 1] - h],
                        [center[0, 0] + w, center[0, 1] + h],
                        [center[0, 0] - w, center[0, 1] + h]
                        ])

    lines = np.array([
        [l_start],
        [l_end]
        ])

    p_lines = np.array([
        polygon,
        np.roll(polygon, 1, axis=0)])

    p_lines_rep = np.reshape(p_lines, (2, -1, 2))
    p_lines_rep = np.tile(p_lines_rep, (len(l_start), 1))

    lines_rep = np.reshape(lines, (2, -1, 2))
    lines_rep = np.repeat(lines_rep, repeats=[len(polygon)], axis=1)

    out = line_line_intersection2(lines_rep, p_lines_rep, closest)
    return out[~np.isnan(out)].reshape((-1, 2))


def line_ellipse_intersection(c_pt, w, h, l_start, l_end, closest=None):
    """Rotation not supported. Can be done with affine rotation"""
    if w > h:
        r = w
        h = h / w
        w = 1
    else:
        r = h
        w = w / h
        h = 1
    affine = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]])
    circle_intersection = line_segment_circle_intersection(c_pt, r, l_start, l_end, closest)
    min_cpt = circle_intersection - c_pt
    dummy = np.tile([1], (len(min_cpt), 1))
    new_pts = np.concatenate((min_cpt, dummy), axis=1)
    out = np.matmul(new_pts, affine)
    return out[..., :2] + c_pt
