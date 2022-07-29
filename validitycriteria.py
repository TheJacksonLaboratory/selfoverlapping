import numpy as np
import matplotlib.pyplot as plt


def get_shifted_indices(a1, a2, n_vertices, exclude_start=True, exclude_end=True):
    shifted_indices = np.mod(a1 + np.arange(n_vertices), n_vertices)
    r = a2 - a1 + 1 + (0 if a2 > a1 else n_vertices)
    return shifted_indices[exclude_start:r - exclude_end]


def get_alias_cut(a1, a2, vertices=None, plot_it=False, left2right=True):
    if vertices is not None:
        left_node = '%s%i%s' % (chr(97 + int(vertices[a1, 2])), int(vertices[a1, -1])+1, '\'' if vertices[a1, -2] < 0.5 else '')
        right_node = '%s%i%s' % (chr(97 + int(vertices[a2, 2])), int(vertices[a2, -1])+1, '\'' if vertices[a2, -2] < 0.5 else '')
        # path = (a1 if left2right else a2, a2 if left2right else a1, left_node + ',' + right_node + (',L' if left2right else ',R'))
        path = (a1, a2, left_node + ',' + right_node + (',L' if left2right else ',R'))

    else:
        path = (a1, a2)

    if plot_it:
        plt.plot(vertices[:, 0], vertices[:, 1], 'b:')
        n_vertices = vertices.shape[0]
        if left2right:
            shifted_indices = np.mod(a1 + np.arange(n_vertices), n_vertices)
            r = a2 - a1 + 1 + (0 if a2 > a1 else n_vertices)
        else:
            shifted_indices = np.mod(a2 + np.arange(n_vertices), n_vertices)
            r = a1 - a2 + 1 + (0 if a1 > a2 else n_vertices)

        plt.plot(vertices[shifted_indices[0:r], 0], vertices[shifted_indices[0:r], 1], 'r-')
        
        ray_ids = vertices[:, 2].astype(np.int32) == int(vertices[a1, 2])
        for x, y, r, s, o in  vertices[ray_ids, ...]:
            plt.text(x, y, '%s%i%s' % (chr(97+int(r)), int(o)+1, '\'' if s < 0.5 else ''), color='c')
            plt.plot(x, y, 'b.')

        if int(vertices[a1, 2]) != int(vertices[a2, 2]):
            ray_ids = vertices[:, 2].astype(np.int32) == int(vertices[a2, 2])
            for x, y, r, s, o in  vertices[ray_ids, ...]:
                plt.text(x, y, '%s%i%s' % (chr(97+int(r)), int(o)+1, '\'' if s < 0.5 else ''), color='c')
                plt.plot(x, y, 'b.')
        plt.title(path)
        plt.show()

    return path


def get_cut(a, vertices, next_cut=True, same_ray=True, sign=0):
    n_vertices = vertices.shape[0]
    ray_a = int(vertices[a, 2])
    
    if same_ray:
        criterion_0 = np.ones(n_vertices, dtype=np.bool)
        criterion_1 = vertices[:, 2].astype(np.int32) == ray_a

        shifted_indices = np.argsort(vertices[:, -1])
        pos_shift = np.where(vertices[shifted_indices, -1].astype(np.int32) == int(vertices[a, -1]))[0][-1 * next_cut]
        shifted_indices = shifted_indices[np.mod(pos_shift + next_cut + np.arange(n_vertices), n_vertices)]
    else:
        shifted_indices = np.mod(a + (1 if next_cut else 0) + np.arange(n_vertices), n_vertices)
        criterion_0 = vertices[:, 2].astype(np.int32) >= 0
        criterion_1 = vertices[:, 2].astype(np.int32) != ray_a
    
    if sign:
        criterion_2 = vertices[:, -2].astype(np.int32) == sign
    else:
        criterion_2 = np.ones(n_vertices, dtype=np.bool)
    
    criteria = criterion_0 * criterion_1 * criterion_2
    b = np.where(criteria[shifted_indices])[0]

    if len(b) == 0:
        return None
    
    b = shifted_indices[b[0 if next_cut else -1]]

    return b


def check_adjacency(a1, a2, vertices, left2right=True):
    """ Check if the cut point a2 is adjacent to the cut a1 to the left/right
    """
    n_vertices = vertices.shape[0]
    
    ray_a1 = int(vertices[a1, 2])
    ray_a2 = int(vertices[a2, 2])

    if left2right:
        shifted_indices = np.mod(a1 + 1 + np.arange(n_vertices), n_vertices)
        r = a2 - a1 - 1 + (0 if a1 < a2 else n_vertices)
    else:
        shifted_indices = np.mod(a2 + 1 + np.arange(n_vertices), n_vertices)
        r = a1 - a2 - 1 + (0 if a2 < a1 else n_vertices)

    # Both points are not adjacent if there is at least one cut point between them
    is_not_adjacent = np.any(
        np.bitwise_and(
                np.bitwise_or(
                    vertices[shifted_indices[:r], 2].astype(np.int32) != ray_a1,
                    vertices[shifted_indices[:r], 2].astype(np.int32) != ray_a2
                ),
                vertices[shifted_indices[:r], 2].astype(np.int32) >= 0
            )
        )
    
    return not is_not_adjacent


def condition_1(a1, a2, vertices, check_left=True, **kwargs):
    children_ids = []
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None, children_ids

    is_valid = check_adjacency(a1, a2, vertices, left2right=check_left)
    if is_valid:
        # children_ids.append([(*get_alias_cut(a1, a2, vertices, left2right=check_left), 1)])
        children_ids.append([(-1, -1, 'self', '1%s' % ('L' if check_left else 'R'))])
    return is_valid, children_ids


def condition_2(a1, a2, vertices, check_left=True, **kwargs):
    children_ids = []
    is_valid = None

    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None, children_ids
    
    b1 = get_cut(a1, vertices, next_cut=check_left, same_ray=False, sign=1)
    b2 = get_cut(a2, vertices, next_cut=not check_left, same_ray=False, sign=-1)

    if b1 is None or b2 is None or not (check_adjacency(a1, b1, vertices, left2right=check_left) and check_adjacency(b2, a2, vertices, left2right=check_left)):
        return None, children_ids
    
    ray_b1 = int(vertices[b1, 2])
    ray_b2 = int(vertices[b2, 2])
    
    if ray_b1 != ray_b2:
        return None, children_ids

    is_valid, child_id = check_validity(b1, b2, vertices, check_left=check_left, **kwargs)
    child_id = (*child_id, '2%s' % ('L' if check_left else 'R'))

    children_ids.append([child_id])
    is_valid = None if isinstance(is_valid, str) and is_valid == 'Explored' else is_valid
    
    return is_valid, children_ids


def condition_3(a1, a2, vertices, check_left=True, **kwargs):
    children_ids = []

    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None, children_ids

    a1_pos = int(vertices[a1, -1])
    a2_pos = int(vertices[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if abs(a1_pos - a2_pos) - 1 < 2:
        return None, children_ids

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    ray_a = int(vertices[a1, 2])
    is_valid = None

    for mc in filter(lambda mc: mc[1] == ray_a, kwargs['min_crest_cuts' if check_left else 'max_crest_cuts']):
        a3 = mc[2 if check_left else 3]
        a4 = mc[3 if check_left else 2]

        # Check that all intersection points are different
        if len(set({a1, a2, a3, a4})) < 4:
            continue

        # Check the order of the intersection vertices on the current ray
        if not (vertices[a1, -1] < vertices[a3, -1] < vertices[a4, -1] < vertices[a2, -1]):
            continue

        # Check if point a3' and a1 are left/right valid
        is_valid_1, child_id_1 = check_validity(a1, a3, vertices, check_left=check_left, **kwargs)
        child_id_1 = (*child_id_1, '3%s' % ('L' if check_left else 'R'))

        # Check if point a2' and a4 are left/right valid
        is_valid_2, child_id_2 = check_validity(a4, a2, vertices, check_left=check_left, **kwargs)
        child_id_2 = (*child_id_2, '3%s' % ('L' if check_left else 'R'))

        children_ids.append([child_id_1, child_id_2])
        is_valid_2 = None if isinstance(is_valid_2, str) and is_valid_2 == 'Explored' else is_valid_2
        is_valid_1 = None if isinstance(is_valid_1, str) and is_valid_1 == 'Explored' else is_valid_1

        if is_valid_1 is None and is_valid_2 is None:
            continue
        elif is_valid_2 is None:
            pair_is_valid = is_valid_1
        elif is_valid_1 is None:
            pair_is_valid = is_valid_2
        else:
            pair_is_valid = is_valid_1 & is_valid_2

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid
        
    return is_valid, children_ids


def condition_4(a1, a2, vertices, check_left=True, **kwargs):
    children_ids = []

    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None, children_ids

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids
    
    a3 = get_cut(a2, vertices, next_cut=True, same_ray=True, sign=1)
    if a3 is None:
        return None, children_ids

    # Verify that a2' and a3 are minimal crest cuts
    if not any(filter(lambda mc: mc[3 if check_left else 2] == a2 and mc[2 if check_left else 3] == a3, kwargs['max_crest_cuts' if check_left else 'min_crest_cuts'])):
        return None, children_ids

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    n_vertices_on_ray = np.max(vertices[vertices[:, 2].astype(np.int32) == int(vertices[a3, 2]), -1])
    set_id = 0

    while int(vertices[a_p, -1]) < n_vertices_on_ray:
        set_id += 1

        a_p = get_cut(a_p, vertices, next_cut=True, same_ray=True, sign=-1)
        if a_p is None:
            break

        # Check if that point and a1 are left/right valid
        is_valid_1, child_id_1 = check_validity(a1, a_p, vertices, check_left=check_left, **kwargs)
        child_id_1 = (*child_id_1, '4%s' % ('L' if check_left else 'R'))

        # Check if that point and a3 are right/left valid
        is_valid_2, child_id_2 = check_validity(a3, a_p, vertices, check_left=not check_left, **kwargs)
        child_id_2 = (*child_id_2, '4%s' % ('L' if check_left else 'R'))

        children_ids.append([child_id_1, child_id_2])

        is_valid_1 = None if isinstance(is_valid_1, str) and is_valid_1 == 'Explored' else is_valid_1
        is_valid_2 = None if isinstance(is_valid_2, str) and is_valid_2 == 'Explored' else is_valid_2

        if is_valid_1 is None and is_valid_2 is None:
            continue
        elif is_valid_2 is None:
            pair_is_valid = is_valid_1
        elif is_valid_1 is None:
            pair_is_valid = is_valid_2
        else:
            pair_is_valid = is_valid_1 & is_valid_2

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid
            
    return is_valid, children_ids


def condition_5(a1, a2, vertices, check_left=True, **kwargs):
    children_ids = []

    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None, children_ids

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    a3 = get_cut(a1, vertices, next_cut=False, same_ray=True, sign=-1)
    if a3 is None:
        return None, children_ids
    
    # Verify that a1 and a3' are minimal crest cuts
    if not any(filter(lambda mc: mc[3 if check_left else 2] == a3 and mc[2 if check_left else 3] == a1, kwargs['max_crest_cuts' if check_left else 'min_crest_cuts'])):
        return None, children_ids

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    set_id = 0
    while int(vertices[a_p, -1]) > 0:
        set_id += 1

        a_p = get_cut(a_p, vertices, next_cut=False, same_ray=True, sign=1)
        if a_p is None:
            break

        # Check if that point and a1 are right valid
        is_valid_1, child_id_1 = check_validity(a_p, a2, vertices, check_left=check_left, **kwargs)
        child_id_1 = (*child_id_1, '5%s' % ('L' if check_left else 'R'))

        # Check if that point and a3 are left valid
        is_valid_2, child_id_2 = check_validity(a_p, a3, vertices, check_left=not check_left, **kwargs)
        child_id_2 = (*child_id_2, '5%s' % ('L' if check_left else 'R'))

        children_ids.append([child_id_1, child_id_2])

        is_valid_1 = None if isinstance(is_valid_1, str) and is_valid_1 == 'Explored' else is_valid_1
        is_valid_2 = None if isinstance(is_valid_2, str) and is_valid_2 == 'Explored' else is_valid_2

        if is_valid_1 is None and is_valid_2 is None:
            continue
        elif is_valid_2 is None:
            pair_is_valid = is_valid_1
        elif is_valid_1 is None:
            pair_is_valid = is_valid_2
        else:
            pair_is_valid = is_valid_1 & is_valid_2

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid, children_ids


def invalid_condition_1(a1, a2, vertices):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None
    
    a1_pos = int(vertices[a1, -1])
    a2_pos = int(vertices[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if a1_pos - a2_pos != 2:
        return None

    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 3:
        return None

    a3 = get_cut(a1, vertices, next_cut=True, same_ray=True, sign=-1)
    
    if a3 is None:
        return None
    
    if not check_adjacency(a3, a1, vertices, left2right=True):
        return None

    return not (vertices[a1, -1] < vertices[a3, -1] < vertices[a2, -1])


def invalid_condition_2(a1, a2, vertices):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None
    
    a1_pos = int(vertices[a1, -1])
    a2_pos = int(vertices[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if a1_pos - a2_pos != 2:
        return None

    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 3:
        return None

    a3 = get_cut(a2, vertices, next_cut=False, same_ray=True, sign=-1)
    
    if a3 is None:
        return None

    if not check_adjacency(a2, a3, vertices, left2right=True):
        return None

    return not (vertices[a1, -1] < vertices[a3, -1] < vertices[a2, -1])


def invalid_condition_3(a1, a2, vertices, tolerance=1e-4):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None
    
    b2 = get_cut(a1, vertices, next_cut=True, same_ray=False, sign=0)
    b1 = get_cut(a2, vertices, next_cut=False, same_ray=False, sign=0)

    if b1 is None or b2 is None:
        return None

    if not (check_adjacency(b1, a2, vertices, left2right=True) and check_adjacency(a1, b2, vertices, left2right=True)):
        return None

    n_vertices = vertices.shape[0]
    # Get all intersections between b2 and a1, and between a2 and b1
    shifted_indices_1 = np.mod(a1 + 1 + np.arange(n_vertices), n_vertices)
    r_1 = (b2 - a1 - 1) + (0 if b2 > a1 else n_vertices)

    shifted_indices_2 = np.mod(b1 + 1 + np.arange(n_vertices), n_vertices)
    r_2 = (a2 - b1 - 1) + (0 if a2 > b1 else n_vertices)
    
    b2_a1_int = list(np.where(vertices[shifted_indices_1[:r_1], 2].astype(np.int32) < -1)[0])
    a2_b1_int = list(np.where(vertices[shifted_indices_2[:r_2], 2].astype(np.int32) < -1)[0])

    # This condition does not apply if only one segment contains any intersection
    if len(b2_a1_int) == 0 or len(a2_b1_int) == 0:
        return None

    int_1 = vertices[shifted_indices_1[b2_a1_int], :2]
    int_1 = int_1 / np.linalg.norm(int_1, axis=1)[..., np.newaxis]
    int_2 = vertices[shifted_indices_2[a2_b1_int], :2]
    int_2 = int_2 / np.linalg.norm(int_2, axis=1)[..., np.newaxis]

    # If there is at least one intersection point in both segments, this cut is right invalid
    return not (np.matmul(int_1, int_2.transpose()) >= 1 - tolerance).any()


all_valid_conditions = [condition_5, condition_4, condition_3, condition_2, condition_1]
all_invalid_conditions = [invalid_condition_1, invalid_condition_2, invalid_condition_3]


def check_validity(a1, a2, vertices, max_crest_cuts, min_crest_cuts, visited, check_left=True):
    cut_id = get_alias_cut(a1, a2, vertices=vertices, plot_it=False, left2right=check_left)

    # Check if this node has been visited before
    is_valid, children_ids = visited.get(cut_id, [None, []])
    
    if is_valid is None:
        visited[cut_id] = ['Explored', children_ids]
        is_valid = True
        
        if check_left:
            for condition in all_invalid_conditions:
                resp = condition(a1, a2, vertices)
                if resp is not None:
                    is_valid &= resp
                if not is_valid:
                    break

        if is_valid:
            # This cut is not left valid until the contrary is proven
            is_valid = None
            for condition in all_valid_conditions:
                resp, cond_children_ids = condition(a1, a2, vertices, max_crest_cuts=max_crest_cuts, min_crest_cuts=min_crest_cuts, visited=visited, check_left=check_left)
                if resp is not None:
                    children_ids += cond_children_ids
                    if is_valid is None:
                        is_valid = resp
                    else:
                        is_valid |= resp
        
        visited[cut_id] = [is_valid, children_ids]

    return is_valid, cut_id


def traverse_tree(root, visited, path=None):
    root = root if len(root) == 3 else root[:-1]

    if path is None:
        path = []
    elif root in path:
        return None
    
    validity, cond_dep = visited.get(root, (True, []))
    if len(cond_dep) == 0:
        return validity

    validity = False
    # Make a copy from the conditions dependency of this node, so the original list can be shrunked if needed
    for sib_cond in list(cond_dep):

        # This path is valid only if all sibling conditions are valid
        sibling_validity = all(map(lambda sib_path: traverse_tree(sib_path, visited, list(path) + [root]), sib_cond))
        
        if not sibling_validity:
            visited[root][1].remove(sib_cond)
        
        validity |= sibling_validity
    
    return validity


def immerse_valid_tree(root, visited, n_vertices, polys_idx):
    """ Traverse only one of the valid immersions for simplicity
    """
    immersion = {root:{}}
    conditions = visited.get(root if len(root) == 3 else root[:-1], None)

    a1, a2 = root[:2]
    left2right = root[2][-1] == 'L'
    sub_poly = []

    if conditions[1][0][0][0] < 0:
        # When a cut that is left/right valid by condition 1, the base case is reached
        if left2right:
            sub_poly.append(get_shifted_indices(a1, a2, n_vertices, exclude_start=False, exclude_end=False))
        else:
            sub_poly.append(get_shifted_indices(a2, a1, n_vertices, exclude_start=False, exclude_end=False))

        return immersion, sub_poly
            
    for sib_cond in conditions[1]:
        if len(sib_cond) == 1:
            child = sib_cond[0]
            child_cond_id = child[-1][0]
            child_left2right = child[-1][1] == 'L'

            sub_immersion, child_poly = immerse_valid_tree(child, visited, n_vertices,polys_idx)
            immersion[root].update(sub_immersion)

            # If this cut was selected by complying with condition 2, add the child path, and send it to the parent path
            b1, b2 = child[:2]
            if child_cond_id == '2':
                if child_left2right:
                    exclude_child_end = False if len(child_poly) == 0 else child_poly[0][0] in [a1, b1]
                    exclude_child_start = False if len(child_poly) == 0 else child_poly[-1][-1] in [a2, b2]
                    
                    sub_poly.append(get_shifted_indices(a1, b1, n_vertices, exclude_start=False, exclude_end=exclude_child_end))
                    sub_poly += child_poly
                    sub_poly.append(get_shifted_indices(b2, a2, n_vertices, exclude_start=exclude_child_start, exclude_end=False))
                else:
                    exclude_child_end = False if len(child_poly) == 0 else child_poly[0][0] in [a2, b2]
                    exclude_child_start = False if len(child_poly) == 0 else child_poly[-1][-1] in [a1, b1]

                    sub_poly.append(get_shifted_indices(a2, b2, n_vertices, exclude_start=False, exclude_end=exclude_child_end))
                    sub_poly += child_poly
                    sub_poly.append(get_shifted_indices(b1, a1, n_vertices, exclude_start=exclude_child_start, exclude_end=False))
            else:
                sub_poly = child_poly

        else:
            child_1, child_2 = sib_cond
            child_cond_id = child_1[-1][0]
            child_left2right = child_1[-1][1] == 'L'

            if child_cond_id == '3':
                sub_immersion, child_poly = immerse_valid_tree(child_1, visited, n_vertices,polys_idx)
                immersion[root].update(sub_immersion)
                polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_immersion, child_poly = immerse_valid_tree(child_2, visited, n_vertices,polys_idx)
                immersion[root].update(sub_immersion)
                polys_idx.append(np.concatenate(child_poly, axis=0))

                if child_left2right:
                    sub_poly.append(get_shifted_indices(child_1[1], child_2[0], n_vertices, exclude_start=False, exclude_end=False))
                else:
                    sub_poly.append(get_shifted_indices(child_2[0], child_1[1], n_vertices, exclude_start=False, exclude_end=False))

            elif child_cond_id == '4':
                sub_immersion, child_poly = immerse_valid_tree(child_1, visited, n_vertices,polys_idx)
                immersion[root].update(sub_immersion)
                sub_poly += child_poly

                if child_left2right:
                    sub_poly.append(get_shifted_indices(child_2[0], a2, n_vertices, exclude_start=False, exclude_end=False))
                else:
                    sub_poly.append(get_shifted_indices(a2, child_2[0], n_vertices, exclude_start=False, exclude_end=False))

                polys_idx.append(np.concatenate(sub_poly, axis=0))

                sub_immersion, child_poly = immerse_valid_tree(child_2, visited, n_vertices,polys_idx)
                immersion[root].update(sub_immersion)
                polys_idx.append(np.concatenate(child_poly, axis=0))
                
                sub_poly = []

            elif child_cond_id == '5':
                sub_immersion, child_poly = immerse_valid_tree(child_1, visited, n_vertices,polys_idx)
                immersion[root].update(sub_immersion)
                sub_poly += child_poly
                
                if child_left2right:
                    sub_poly.append(get_shifted_indices(a1, child_2[1], n_vertices, exclude_start=False, exclude_end=False))
                else:
                    sub_poly.append(get_shifted_indices(child_2[1], a1, n_vertices, exclude_start=False, exclude_end=False))

                polys_idx.append(np.concatenate(sub_poly, axis=0))

                sub_immersion, child_poly = immerse_valid_tree(child_2, visited, n_vertices,polys_idx)
                immersion[root].update(sub_immersion)
                polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_poly = []
        break   

    return immersion, sub_poly

    
def discover_polygons(polys_idx, vertices):
    polys = []
    n_vertices = vertices.shape[0]

    for poly_ids in polys_idx:
        polys.append(vertices[poly_ids, :2])

    return polys