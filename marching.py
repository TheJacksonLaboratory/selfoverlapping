import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import blob

def compute_area(poly):
    """
    Compute the signed area of the polygon
    """
    poly = np.vstack((poly, poly[0, ...]))
    area = np.sum(poly[:-1, 0] * poly[1:, 1] - poly[:-1, 1] * poly[1:, 0]) / 2.0
    return area


def find_intersections(vertices, rays_formulae, using_rays=False):
    """
    Find all intersection points between the rays and the polygon edges
    """
    n_vertices = vertices.shape[0]

    # Find the cuts made by each ray to each line defined between two points in the polygon
    all_vecs = []
    all_mats = []
    valid_edges = []  
    for i in range(n_vertices):
        j = (i + 1) % n_vertices

        vx = vertices[j, 0] - vertices[i, 0]
        vy = vertices[j, 1] - vertices[i, 1]

        px = vertices[i, 0]
        py = vertices[i, 1]

        # Only add non-singular equation systems i.e. edges that are actually crossed by each ray
        for k, ((rs_x, rs_y, _), (rd_x, rd_y, _)) in enumerate(rays_formulae):
            if not using_rays and (i == k or j == k):
                continue
            mat = np.array([[vx, -rd_x], [vy, -rd_y]])
            vec = np.array([rs_x - px, rs_y - py])

            if math.fabs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) < 1e-3:
                continue

            all_mats.append(mat)
            all_vecs.append(vec)

            valid_edges.append((i, j, k))

    all_mats = np.stack(all_mats)
    all_vecs = np.stack(all_vecs)

    t_coefs = np.linalg.solve(all_mats, all_vecs)
    valid_edges = np.array(valid_edges, dtype=np.int32)
    
    return t_coefs, valid_edges



def compute_valid_cuts(vertices, t_coefs, valid_edges, using_rays=False, tolerance=1e-8):
    """
    Filter only valid cuts. A cut is valid if the coeficient lies between 0 and 1
    """
    if using_rays:
        t_valids = np.bitwise_and(tolerance <= t_coefs[:, 0], t_coefs[:, 0] <= 1.0-tolerance)
    else:
        t_valids = np.bitwise_and(np.bitwise_and(tolerance <= t_coefs[:, 0], t_coefs[:, 0] <= 1.0-tolerance), np.bitwise_and(tolerance <= t_coefs[:, 1], t_coefs[:, 1] <= 1.0-tolerance))

    i_valids = valid_edges[t_valids, 0]
    j_valids = valid_edges[t_valids, 1]
    r_valids = valid_edges[t_valids, 2]

    t_coefs = t_coefs[t_valids, 0]

    vx = vertices[j_valids, 0] - vertices[i_valids, 0]
    vy = vertices[j_valids, 1] - vertices[i_valids, 1]

    px = vertices[i_valids, 0]
    py = vertices[i_valids, 1]

    # Store the new cut vertex
    cut_x = px + vx * t_coefs
    cut_y = py + vy * t_coefs

    cut_x = cut_x.reshape(-1, 1)
    cut_y = cut_y.reshape(-1, 1)

    return (cut_x, cut_y), (i_valids, j_valids, r_valids), t_coefs
    

def merge_new_vertices(vertices, cut_x, cut_y, t_coefs, i_valids, r_valids):
    """
    Merge the new computed vertices into the existing polygon
    """
    n_vertices = vertices.shape[0]
    n_new_vertices = 0
    new_vertices = []
    new_indices = []
    for i in range(n_vertices):
        new_vertices.append([vertices[i, 0], vertices[i, 1], -1])
        n_new_vertices += 1

        sel_i = np.nonzero(i_valids == i)[0]
        sel_r = r_valids[sel_i]
        if len(sel_i) == 0:
            continue

        ord_idx = np.argsort(t_coefs[sel_i])
        
        sel_i = sel_i[ord_idx]
        sel_r = sel_r[ord_idx]

        new_vertices.append(np.hstack((cut_x[sel_i], cut_y[sel_i], sel_r.reshape(-1, 1))))
        new_indices += list(range(n_new_vertices, n_new_vertices + len(ord_idx)))
        n_new_vertices += len(ord_idx)

    new_vertices = np.vstack(new_vertices)
    return new_vertices, new_indices


def compute_cut_points(vertices, rays_formulae, using_rays=True):
    """
    Compute all intersection points between the polygon edges and the rays passed
    """
    t_coefs, valid_edges = find_intersections(vertices, rays_formulae, using_rays)

    (cut_x, cut_y), (i_valids, _, r_valids), t_coefs = compute_valid_cuts(vertices, t_coefs, valid_edges, using_rays)
    
    new_vertices, new_indices = merge_new_vertices(vertices, cut_x, cut_y, t_coefs, i_valids, r_valids)

    return new_vertices, new_indices


def all_self_intersections(vertices):
    """
    Compute a ray for each edge in the polygon
    """
    n_vertices = vertices.shape[0]
    rays_formulae = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        px, py = vertices[i, :]
        qx, qy = vertices[j, :]
        vx, vy = qx - px, qy - py
        
        rays_formulae.append(((px, py, 1), (vx, vy, 0)))

    return rays_formulae


def discover_polygons(vertices):
    """
    Find sub-polygons using the marching algorithm (suggested by Jim Peterson in slack).
    A new polygon is subtracted from the parent polygon if there is a path that begins and ends in the same intesection cut.
    """

    # Get all intersection points
    all_intersection_ids = list(np.where(vertices[:, 2].astype(np.int32) != -1)[0])

    # Look for loops on the polygon
    for id in all_intersection_ids:
        # When the first point that generates a loop is found, exit this cycle        
        e_id = list(np.where(np.sum(np.abs(vertices[(id+1):, :2] - vertices[id, :2]), axis=1) <= 1e-8)[0])
        if len(e_id) > 0:
            e_id = e_id[0] + id + 1
            s_id = id
            break
    else:
        # No more loops were found in this sub-polygon
        return [vertices[:, :2]]
    
    n_vertices = vertices.shape[0]
    rem_ids = np.setdiff1d(np.arange(n_vertices), np.arange(s_id, e_id))

    # Discover polygons inside the extracted one
    new_poly = []
    vertices[s_id, 2] = -1
    vertices[e_id, 2] = -1
    new_poly += discover_polygons(vertices[s_id:e_id, ...])

    # Discover polygons on the remaining vertices
    new_poly += discover_polygons(vertices[rem_ids, ...])

    return new_poly


def check_holes(polys):
    """
    Compute the signed area of each polygon.
    Holes will have an area with the opposite sign.
    """
    areas = list(map(compute_area, polys))
    max_area_id = np.argmax(np.fabs(areas))
    max_area = areas[max_area_id]
    max_area_symbol = 1 if max_area > 0.0 else -1

    valid_ids = list(map(lambda i_a: i_a[0], filter(lambda i_a: max_area_symbol * i_a[1] > 0, enumerate(areas))))
    holes_ids = list(np.setdiff1d(range(len(polys)), valid_ids))

    valid_polys = [polys[v] for v in valid_ids]
    holes = [polys[h] for h in holes_ids]

    return valid_polys, holes


if __name__ == '__main__':
    # vertices = np.array([[[20,20],[60,20],[60,60],[20,60],[10,50],[10,30],[30,30],[15,35],[20,40]]])*5
    vertices = blob.mouse()

    # Check that the vertices are given in clockwise ordering
    rays_formulae = all_self_intersections(vertices[0])

    # Compute and merge the self-intersections with the original vertices
    new_vertices, new_indices = compute_cut_points(vertices[0], rays_formulae, using_rays=False)
    
    # New vertices has in its third column a flag indicating if the vertex is a self-intersection (!= -1), or a normal vertex (-1)
    # The value on this third column is the index of the edge where it this intersectio blongs.

    im = np.zeros([700, 700, 3], dtype=np.uint8)
    
    cv2.drawContours(im, [vertices], 0, (255, 255, 255), 1)

    for x, y in new_vertices[new_vertices[:, 2].astype(np.int32) != -1, :2]:
        cv2.circle(im, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    polys = discover_polygons(new_vertices)
    valid_polys, holes = check_holes(polys)

    for p, poly in enumerate(valid_polys):
        color = int((p+1) / len(valid_polys) * 255.0)
        color = (color, color, color)
        cv2.drawContours(im, [poly.astype(np.int32)], 0, color, -1)
    
    for hole in holes:
        cv2.drawContours(im, [hole.astype(np.int32)], 0, (0, 0, 0), -1)
    
    cv2.imshow('Filled poly', im)
    cv2.waitKey()
    cv2.destroyAllWindows()
