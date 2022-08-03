import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import math
import numpy as np

import cv2
import test_polys
from functools import reduce

from validitycriteria import check_validity, traverse_tree, immerse_valid_tree, discover_polygons


def _get_root_indices(new_vertices):
    """ Find the indices of the vertices that define the root cut used to recurse the polygon subdivision algorithm.
    
    Parameters 
    ----------
    new_vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices and additional information about each.
    
    Returns
    -------
    left_idx : int
        The positional index of the root left vertex
    right_idx : int
        The positional index of the root right vertex

    """
    n_vertices = new_vertices.shape[0]

    # Non-intersection points:
    org_ids = np.where(new_vertices[:, 2].astype(np.int32) == -1)[0]

    root_vertex = org_ids[np.argmax(new_vertices[org_ids, 1])]

    shifted_indices = np.mod(root_vertex + 1 + np.arange(n_vertices), n_vertices)
    right_idx = np.where(new_vertices[shifted_indices, 2].astype(np.int32) == 0)[0][0]
    right_idx = shifted_indices[right_idx]

    shifted_indices = np.mod(root_vertex + np.arange(n_vertices), n_vertices)
    left_idx = np.where(new_vertices[shifted_indices, 2].astype(np.int32) == 0)[0][-1]
    left_idx = shifted_indices[left_idx]

    return left_idx, right_idx


def _get_crest_ids(vertices):
    """ Finds the positional indices of the crest points.
    Only crests where there is a left turn on the polygon perimeter are considered.
    
    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    
    Returns
    -------
    max_crest_ids : list
        A list of indices of the maximum crest vertices.
    min_crest_ids : list
        A list of indices of the minimum crest vertices.
    max_crest : int
        The positional index of the maximum crest vertex.
    min_crest : int
        The positional index of the minimum crest vertex.

    """
    n_vertices = vertices.shape[0]

    max_crest_ids = []
    min_crest_ids = []
    climbing_up = False
    climbing_down = False

    last_candidate = None

    for i in range(n_vertices):
        prev_idx = (i - 1) % n_vertices
        next_idx = (i + 1) % n_vertices
        
        direction = vertices[next_idx, 0] - vertices[prev_idx, 0] < 0

        higher_than_prev = vertices[i, 1] > vertices[prev_idx, 1]
        higher_than_next = vertices[i, 1] > vertices[next_idx, 1]
        lower_than_prev = vertices[i, 1] < vertices[prev_idx, 1]
        lower_than_next = vertices[i, 1] < vertices[next_idx, 1]

        equal_to_prev = vertices[i, 1] == vertices[prev_idx, 1]
        equal_to_next = vertices[i, 1] == vertices[next_idx, 1]
        
        if higher_than_prev and higher_than_next:
            # Add this point only if the curve is turning left
            if direction:
                max_crest_ids.append(i)
            climbing_up = False
            last_candidate = None

        elif lower_than_prev and lower_than_next:
            if not direction:
                min_crest_ids.append(i)
            climbing_down = False
            last_candidate = None

        elif equal_to_prev and higher_than_next:
            if climbing_up and direction:
                max_crest_ids.append(last_candidate)
            climbing_up = False
            climbing_down = False
            last_candidate = None
        
        elif equal_to_prev and lower_than_next:
            if climbing_down and not direction:
                min_crest_ids.append(last_candidate)
            climbing_up = False
            climbing_down = False
            last_candidate = None

        elif higher_than_prev and equal_to_next:
            climbing_up = True
            last_candidate = i

        elif lower_than_prev and equal_to_next:
            climbing_down = True
            last_candidate = i
    
    max_crest = reduce(lambda v1, v2: v1 if v1[0] > v2[0] else v2, map(lambda idx: (vertices[idx, 1], idx), max_crest_ids), (-float('inf'), None))[1]
    min_crest = reduce(lambda v1, v2: v1 if v1[0] < v2[0] else v2, map(lambda idx: (vertices[idx, 1], idx), min_crest_ids), (float('inf'), None))[1]

    return max_crest_ids, min_crest_ids, max_crest, min_crest


def _get_crest_cuts(new_vertices, crest_ids):    
    """ Finds the positional indices of the intersection vertices closest to each crest point.

    Parameters 
    ----------
    new_vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices with additional information of each vertex.
    
    Returns
    -------
    closest_ids : list
        A list of tuples with the index of the crest point, the positional index of the two intersection vertices
        that are closest to that, and their corresponding ray index. 

    """
    n_vertices = new_vertices.shape[0]
    closest_ids = []

    for i in crest_ids:
        shifted_indices = np.mod(i + np.arange(n_vertices), n_vertices)
        prev_id = np.where(new_vertices[shifted_indices, 2].astype(np.int32) >= 0)[0][-1]
        prev_id = shifted_indices[prev_id]

        shifted_indices = np.mod(i + 1 + np.arange(n_vertices), n_vertices)
        next_id = np.where(new_vertices[shifted_indices, 2].astype(np.int32) >= 0)[0][0]
        next_id = shifted_indices[next_id]

        r = int(new_vertices[prev_id, 2])

        closest_ids.append((i, r, prev_id, next_id))

    return closest_ids


def _get_self_intersections(vertices):    
    """ Computes the rays formulae of all edges to identify self-intersections later.

    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    
    Returns
    -------
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray for each edge on the polygon.
        The tuple contains the source coordinate and the vectorial direction of the ray.
        It also states that the rays were not computed from a crest point (last element in the tuple = False).        

    """
    n_vertices = vertices.shape[0]
    rays_formulae = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        px, py = vertices[i, :]
        qx, qy = vertices[j, :]
        vx, vy = qx - px, qy - py
        
        rays_formulae.append(((px, py, 1), (vx, vy, 0), False))

    return rays_formulae


def _compute_rays(vertices, crest_ids, epsilon=1e-1):
    """ Computes the rays that cross each crest point, offsetted by epsilon.
    The rays are returned in general line form.
    For maximum crests, give a negative epsion instead.

    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    crest_ids : list
        A list of positional indices that correspond to maximum/minimum crest vertices.
    epsilon : float, optional
        An offset added to the crest point in the y-axis.
    
    Returns
    -------
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray for each crest point.
        The tuple contains the source coordinate and the vectorial direction of the ray.
        It also states that the rays were computed from a crest point (last element in the tuple = True).        

    """
    rays_formulae = []
    existing_heights = []
    for ids in crest_ids:
        ray_src = np.array((vertices[ids, 0] - 1.0, vertices[ids, 1] + epsilon, 1))
        ray_dir = np.array((1.0, 0.0, 0.0))
        if len(existing_heights) == 0:
            existing_heights.append(ray_src[1])
            rays_formulae.append((ray_src, ray_dir, True))
            
        elif ray_src[1] not in existing_heights:
            rays_formulae.append((ray_src, ray_dir, True))
    
    return rays_formulae


def _sort_rays(rays_formulae, max_crest_y, tolerance=1e-8):
    """ Sorts the rays according to its relative position to the crest point.
    It also filters any repeated rays.

    Parameters 
    ----------
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray for each crest point.
    max_crest_y : float
        The coordinate in the y-axis of the topmost crest vertex.
    tolerance : float, optional
        The tolerance used to remove any ray that is at less distance than `toelrance` from any other existing ray.
    
    Returns
    -------
    sorted_rays_formulae : list
        The list of unique rays formulae sorted according to their position in the y-axis.        

    """
    if len(rays_formulae) == 1:
        return rays_formulae
    
    sorted_rays_formulae = sorted(map(lambda ray: (math.fabs(ray[0][1] - max_crest_y), ray), rays_formulae))
    sorted_rays_formulae = list(map(lambda ray_tuple: ray_tuple[1], sorted_rays_formulae))

    curr_id = len(sorted_rays_formulae) - 1
    while curr_id > 0:
        curr_y = sorted_rays_formulae[curr_id][0][1]
        same_ray = any(filter(lambda r:  math.fabs(r[0][1] - curr_y) < tolerance, sorted_rays_formulae[:curr_id]))
        # If there is at least one ray close (< tolerance) to this, remove the current ray
        if same_ray:
            sorted_rays_formulae.pop(curr_id)
        curr_id -= 1

    return sorted_rays_formulae


def _find_intersections(vertices, rays_formulae, tolerance=1e-8):
    """ Walks the polygon to find self-intersections and intersections with drawn rays.
    
    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray.
    tolerance : float, optional
        The tolerance used to determine if a ray iside an edge of the polygon.
    
    Returns
    -------
    cut_coords : numpy.ndarray
        A two-dimensional array with coordinates of the intersection vertices.
    valid_edges : numpy.ndarray
        A two-dimensional array with the information of the edges that were cut by a ray.
    t_coefs : numpy.ndarray
        The coefficients of the parametric rays that define the position of the intersection vertex on its respective edge.
    """
    n_vertices = vertices.shape[0]

    # Find the cuts made by each ray to each line defined between two points in the polygon
    valid_edges = []
    valid_cuts = []  
    valid_t_coefs = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices

        vx = vertices[j, 0] - vertices[i, 0]
        vy = vertices[j, 1] - vertices[i, 1]

        px = vertices[i, 0]
        py = vertices[i, 1]

        # Only add non-singular equation systems i.e. edges that are actually crossed by each ray
        for k, ((rs_x, rs_y, _), (rd_x, rd_y, _), is_ray) in enumerate(rays_formulae):
            if not is_ray and (i == k or j == k):
                continue

            mat = np.array([[vx, -rd_x], [vy, -rd_y]])
            vec = np.array([rs_x - px, rs_y - py])

            if math.fabs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) < 1e-3:
                continue

            # Find the parameter `t` that defines the intersection between the current polygon edge and the testing ray
            t_coefs = np.linalg.solve(mat, vec)

            # Determine if the intersection is inside the current edge or not
            if is_ray:
                inter_in_edge = tolerance <= t_coefs[0] <= 1.0-tolerance
            else:
                inter_in_edge = tolerance <= t_coefs[0] <= 1.0-tolerance and tolerance <= t_coefs[1] <= 1.0-tolerance

            # Add this cut if it is on an edge of the polygon
            if inter_in_edge:
                valid_edges.append((i, j, k if is_ray else -2))
                valid_cuts.append((px + vx * t_coefs[0], py + vy * t_coefs[0]))
                valid_t_coefs.append(t_coefs[0])

    valid_edges = np.array(valid_edges, dtype=np.int32)
    cut_coords = np.array(valid_cuts)
    t_coefs = np.array(valid_t_coefs)

    return cut_coords, valid_edges, t_coefs


def _sort_ray_cuts(new_vertices, rays_formulae):
    """ Adds the positional ordering of the intersection vertices that are on a ray.
    It also assigns their corresponding sign according to the direction of the polygon when it is walked from left to right.
    
    Parameters 
    ----------
    new_vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices with additional information about each vertex.
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray.
    
    Returns
    -------
    new_vertices : numpy.ndarray
        The set of vetices coordinates with the updated information about the position of intersection vertices.

    """
    all_idx_per_ray = []
    all_ord_per_ray = []
    all_sym_per_ray = []
    for k, ((_, rs_y, _), _, _) in enumerate(rays_formulae):
        sel_k = np.nonzero(new_vertices[:, -1].astype(np.int32) == k)[0]
        if len(sel_k) == 0:
            continue
        
        # determine the intersection's symbol using the y coordinate of the previous point of each intersection (sel_k - 1)
        inter_symbols = (new_vertices[sel_k - 1, 1] < rs_y) * 2 - 1
        
        rank_ord = np.empty(len(sel_k))
        rank_ord[np.argsort(new_vertices[sel_k, 0])] = list(range(len(sel_k)))

        all_idx_per_ray += list(rank_ord)
        all_ord_per_ray += list(sel_k)
        all_sym_per_ray += list(inter_symbols)

    new_vertices = np.hstack((new_vertices, np.zeros((new_vertices.shape[0], 2))))
    # Fourth column contains the symbol of the cut, and the sixth column the index of that cut on the corresponding ray
    new_vertices[all_ord_per_ray, -2] = all_sym_per_ray
    new_vertices[all_ord_per_ray, -1] = all_idx_per_ray

    return new_vertices


def _merge_new_vertices(vertices, cut_coords, valid_edges, t_coefs):
    """ Merges the new vertices computed from self-intersections and intersections of the polygon with any ray.
    The newly inserted vertices are sorted according to the coefficient used to compute them.

    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    cut_coords : numpy.ndarray
        A two-dimensional array with coordinates of the intersection vertices.
    valid_edges : numpy.ndarray
        A two-dimensional array with the information of the edges that were cut by a ray.
    t_coefs : numpy.ndarray
        The coefficients of the parametric rays that define the position of the intersection vertex on its respective edge.
    
    Returns
    -------
    new_vertices : numpy.ndarray
        The set of vetices coordinates with the updated information about the position of intersection vertices.

    """
    new_vertices = []
    last_j = 0

    for i in np.unique(valid_edges[:, 0]):
        new_vertices.append(np.hstack((vertices[last_j:i+1, :], -np.ones((i-last_j+1, 1)))))
        sel_i = np.nonzero(valid_edges[:, 0] == i)[0]
        sel_r = valid_edges[sel_i, 2]

        ord_idx = np.argsort(t_coefs[sel_i])
        
        sel_i = sel_i[ord_idx]
        sel_r = sel_r[ord_idx]

        new_vertices.append(np.hstack((cut_coords[sel_i], sel_r.reshape(-1, 1))))
        last_j = i + 1

    n_vertices = vertices.shape[0]
    new_vertices.append(np.hstack((vertices[last_j:, :], -np.ones((n_vertices-last_j, 1)))))
    new_vertices = np.vstack(new_vertices)

    return new_vertices


def polygon_subdivision(vertices):
    """ Divide a self-overlapping polygon into non self-overlapping polygons.
    This implements the algorithm proposed in [1].

    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    
    Returns
    -------
    sub_polys : list
        A list of numpy.ndarrays with the coordinates of the non self-overlapping polygons obtained from the original polygon.

    References
    ----------
    .. [1] Uddipan Mukherjee. (2014). Self-overlapping curves:
           Analysis and applications. Computer-Aided Design, 46, 227-232.
           :DOI: https://doi.org/10.1016/j.cad.2013.08.037
    """
    max_crest_ids, min_crest_ids, max_crest, min_crest = _get_crest_ids(vertices)
    
    if max_crest is None and min_crest is None:
        raise ValueError("No crest points found! The polygon is not self-overlapping")

    rays_max = _compute_rays(vertices, max_crest_ids, -0.1)
    rays_min = _compute_rays(vertices, min_crest_ids, 0.1)

    rays_formulae = _sort_rays(rays_max + rays_min, vertices[:, 1].max())
    self_inter_formulae = _get_self_intersections(vertices)

    cut_coords, valid_edges, t_coefs = _find_intersections(vertices, rays_formulae + self_inter_formulae)
    new_vertices = _merge_new_vertices(vertices, cut_coords, valid_edges, t_coefs)
    new_vertices = _sort_ray_cuts(new_vertices, rays_formulae)

    # Get the first point at the left of the crest point
    new_max_crest_ids, new_min_crest_ids, _, _ = _get_crest_ids(new_vertices)
    new_max_crest_cuts = _get_crest_cuts(new_vertices, new_max_crest_ids)
    new_min_crest_cuts = _get_crest_cuts(new_vertices, new_min_crest_ids)
    
    left_idx, right_idx = _get_root_indices(new_vertices)

    # The root is left valid by construction.
    # Therefore, the right validity of the root cut is checked and then all the possible valid cuts are computed.
    visited = {}
    _, root_id = check_validity(left_idx, right_idx, new_vertices, new_max_crest_cuts, new_min_crest_cuts, visited, check_left=False)

    # Update the visited dictionary to leave only valid paths
    _ = traverse_tree(root_id, visited)

    # Perform a single immersion on the validity tree to get the first valid path that cuts the polygon into non self-overlapping sub-polygons
    polys_idx = []
    _, sub_poly = immerse_valid_tree(root_id, visited, new_vertices.shape[0], polys_idx)

    # Add the root cut of the immersion tree
    n_vertices = new_vertices.shape[0]
    shifted_indices = np.mod(left_idx + np.arange(n_vertices), n_vertices)
    r = right_idx - left_idx + 1 + (0 if right_idx > left_idx else n_vertices)
    sub_poly = [shifted_indices[:r]] + sub_poly
    polys_idx.insert(0, np.concatenate(sub_poly, axis=0))
    
    polys = discover_polygons(polys_idx, new_vertices)

    if len(polys) == 0:
        raise ValueError("Couldn\'t split main poly ... check if it is actually self-overlapping")

    return polys


if __name__ == '__main__':
    for poly_test in [test_polys.test_1, test_polys.test_2, test_polys.test_3, test_polys.test_4, test_polys.test_5, test_polys.test_6, test_polys.test_7, test_polys.test_8, test_polys.test_9, test_polys.test_10]:
        vertices = poly_test()

        try:
            polys = polygon_subdivision(vertices[0])

            fig, ax = plt.subplots()
            
            patches = []
            for id, poly in enumerate(polys):
                patches.append(Polygon(poly, True))

            ax.plot(vertices[0][:, 0], vertices[0][:, 1], 'b-')
            ax.plot([vertices[0][-1, 0], vertices[0][0, 0]], [vertices[0][-1, 1], vertices[0][0, 1]], 'b-')

            colors = 100 * np.random.rand(len(polys))
            p = PatchCollection(patches, alpha=0.5)
            p.set_array(colors)
            ax.add_collection(p)
            
            plt.show()
        except ValueError:
            print('Polygon is not self-overlapping')
        except IndexError:
            print('Polygon is not self-overlapping')
        
        polys = vertices
        
        fig, ax = plt.subplots()
        ax.plot(vertices[0][:, 0], vertices[0][:, 1], 'b-')
        ax.plot([vertices[0][-1, 0], vertices[0][0, 0]], [vertices[0][-1, 1], vertices[0][0, 1]], 'b-')

        patches = []
        for poly in polys:
            patches.append(Polygon(poly, True))
            
        colors = 100 * np.random.rand(len(polys))
        p = PatchCollection(patches, alpha=0.5)
        p.set_array(colors)
        ax.add_collection(p)
        
        plt.show()

        im = np.zeros([700, 700, 3], dtype=np.uint8)
        
        cv2.drawContours(im, vertices, 0, (127, 127, 127), -1)
        
        for x, y in vertices[0]:
            cv2.circle(im, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        for p, poly in enumerate(polys):
            color = int((p+1) / len(polys) * 255.0)
            color = (color, color, color)
            cv2.drawContours(im, [poly.astype(np.int32)], 0, color, -1)
            
        cv2.drawContours(im, vertices, 0, (255, 0, 0), 1)
        
        cv2.imshow('Filled poly', im)
        cv2.waitKey()
        cv2.destroyAllWindows()
