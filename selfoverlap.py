import matplotlib.pyplot as plt
import math
import numpy as np

import blob
from functools import reduce

from validitycriteria import check_validity


def get_root_indices(vertices):
    n_vertices = vertices.shape[0]

    # Non-intersection points:
    org_ids = np.where(vertices[:, 2].astype(np.int32) == -1)[0]

    root_vertex = org_ids[np.argmax(vertices[org_ids, 1])]

    shifted_indices = np.mod(root_vertex + 1 + np.arange(n_vertices), n_vertices)
    right_idx = np.where(vertices[shifted_indices, 2].astype(np.int32) == 0)[0][0]
    right_idx = shifted_indices[right_idx]

    shifted_indices = np.mod(root_vertex + np.arange(n_vertices), n_vertices)
    left_idx = np.where(vertices[shifted_indices, 2].astype(np.int32) == 0)[0][-1]
    left_idx = shifted_indices[left_idx]

    return left_idx, right_idx


def get_crestpoints(vertices):
    # Vertices must be a 2D numpy array. First column is the x axis, and secod the y axis of each vertex coordinate
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
    
    max_crest = reduce(lambda v1, v2: v1 if v1[0] > v2[0] else v2, map(lambda idx: (vertices[idx, 1], idx), max_crest_ids), (-float('inf'), None))
    min_crest = reduce(lambda v1, v2: v1 if v1[0] < v2[0] else v2, map(lambda idx: (vertices[idx, 1], idx), max_crest_ids), (float('inf'), None))

    return max_crest_ids, min_crest_ids, max_crest[1], min_crest[1]


def get_crest_cuts(vertices, crest_ids):
    n_vertices = vertices.shape[0]
    closest_id = []

    for i in crest_ids:
        shifted_indices = np.mod(i + np.arange(n_vertices), n_vertices)
        prev_id = np.where(vertices[shifted_indices, 2].astype(np.int32) >= 0)[0][-1]
        prev_id = shifted_indices[prev_id]

        shifted_indices = np.mod(i + 1 + np.arange(n_vertices), n_vertices)
        next_id = np.where(vertices[shifted_indices, 2].astype(np.int32) >= 0)[0][0]
        next_id = shifted_indices[next_id]

        r = int(vertices[prev_id, 2])

        closest_id.append((i, r, prev_id, next_id))

    return closest_id


def compute_rays(vertices, crest_ids, epsilon=1e-1):
    """
    Compute the rays that cross each crest point, offsetted by epsilon.
    The rays are returned in general line form.
    For maximum crests, give a negative epsion instead.
    """
    rays_formulae = []
    for ids in crest_ids:
        ray_src = np.array((vertices[ids, 0] - 1.0, vertices[ids, 1] + epsilon, 1))
        ray_dir = np.array((1.0, 0.0, 0.0))
        rays_formulae.append((ray_src, ray_dir))
    
    return rays_formulae


def sort_rays(rays_formulae, max_crest_y, tolerance=1e-8):
    """
    Sort the rays according to its relative position to the crest poin.
    Filter here any repeated ray
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


def find_intersections(vertices, rays_formulae, using_rays=False):
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
    # Filter only coeficients between 0 and 1
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


def sort_ray_cuts(new_vertices, rays_formulae, direction=1.0):
    # Store the symbol of the intersection for that specific ray according to the direction of the curve between those two vertices
    # Also store the position in the curve according to the original indexing to keep track of adjacency

    # Re-arrange the crossings per ray
    all_idx_per_ray = []
    all_ord_per_ray = []
    all_sym_per_ray = []
    for k, ((_, rs_y, _), _) in enumerate(rays_formulae):
        sel_k = np.nonzero(new_vertices[:, -1].astype(np.int32) == k)[0]
        if len(sel_k) == 0:
            continue
        
        # determine the intersection's symbol using the y coordinate of the previous point of each intersection (sel_k - 1)
        int_symbols = (direction * new_vertices[sel_k - 1, 1] > direction * rs_y) * 2 - 1
        
        rank_ord = np.empty(len(sel_k))
        rank_ord[np.argsort(new_vertices[sel_k, 0])] = list(range(len(sel_k)))

        all_idx_per_ray += list(rank_ord)
        all_ord_per_ray += list(sel_k)
        all_sym_per_ray += list(int_symbols)

    new_vertices = np.hstack((new_vertices, np.zeros((new_vertices.shape[0], 2))))
    new_vertices[all_ord_per_ray, -2] = all_sym_per_ray
    new_vertices[all_ord_per_ray, -1] = all_idx_per_ray

    return new_vertices


def merge_new_vertices(vertices, cut_x, cut_y, t_coefs, i_valids, r_valids):
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


def compute_cut_points(vertices, rays_formulae, direction=1.0, using_rays=True):
    t_coefs, valid_edges = find_intersections(vertices, rays_formulae, using_rays)

    (cut_x, cut_y), (i_valids, _, r_valids), t_coefs = compute_valid_cuts(vertices, t_coefs, valid_edges, using_rays)
    
    new_vertices, new_indices = merge_new_vertices(vertices, cut_x, cut_y, t_coefs, i_valids, r_valids)
    new_vertices = sort_ray_cuts(new_vertices, rays_formulae, direction)

    return new_vertices, new_indices


def all_self_intersections(vertices):
    n_vertices = vertices.shape[0]
    rays_formulae = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        px, py = vertices[i, :]
        qx, qy = vertices[j, :]
        vx, vy = qx - px, qy - py
        
        rays_formulae.append(((px, py, 1), (vx, vy, 0)))

    return rays_formulae
    

def merge_inter(new_vertices, sint_vertices, new_indices):
    old_indices = np.setdiff1d(range(sint_vertices.shape[0]), new_indices)
    sint_vertices[old_indices, ...] = new_vertices

    # Intersections will be marked as -2 in the third columns
    sint_vertices[new_indices, 2] = -2
    return sint_vertices
    

def poly_subdivision(vertices):
    max_crest_ids, min_crest_ids, max_crest, min_crest = get_crestpoints(vertices)
    
    rays_max = compute_rays(vertices, max_crest_ids, -0.1)
    rays_min = compute_rays(vertices, min_crest_ids, 0.1)

    rays_formulae = sort_rays(rays_max + rays_min, vertices[max_crest, 1])
    
    new_vertices, _ = compute_cut_points(vertices, rays_formulae, direction=-1.0, using_rays=True)

    sint_formulae = all_self_intersections(vertices)
    sint_vertices, new_indices = compute_cut_points(new_vertices, sint_formulae, direction=-1.0, using_rays=False)

    new_vertices = merge_inter(new_vertices, sint_vertices, new_indices)

    # Get the first point at the left of the crest point
    new_max_crest_ids, new_min_crest_ids, new_max_crest, new_min_crest = get_crestpoints(new_vertices)
    new_max_crest_cuts = get_crest_cuts(new_vertices, new_max_crest_ids)
    new_min_crest_cuts = get_crest_cuts(new_vertices, new_min_crest_ids)
    
    left_idx, right_idx = get_root_indices(new_vertices)

    # The root is then the right validity check of this cut
    visited = check_validity(left_idx, right_idx, new_vertices, new_max_crest_cuts, new_min_crest_cuts)

    print(visited['validity'])
    print(visited['validity_tree'])

    plt.plot(new_vertices[:, 0], new_vertices[:, 1], 'b-')
    for (_, rs_y, _), _ in rays_formulae:
        plt.plot([np.min(vertices[:, 0]) - 10, np.max(vertices[:, 0]) + 10], [rs_y, rs_y], 'c:')
    plt.show()


if __name__ == '__main__':
    vertices = blob.blob()
    poly_subdivision(vertices[0])
