import numpy as np
import matplotlib.pyplot as plt


def get_alias_cut(a1, a2, vertices=None, plot_it=False, left2right=True):
    if vertices is not None:
        left_node = '%s%i%s' % (chr(97 + int(vertices[a1, 2])), int(vertices[a1, -1])+1, '\'' if vertices[a1, -2] < 0.5 else '')
        right_node = '%s%i%s' % (chr(97 + int(vertices[a2, 2])), int(vertices[a2, -1])+1, '\'' if vertices[a2, -2] < 0.5 else '')
        path = left_node + ',' + right_node + (',L' if left2right else ',R')

    elif left2right:
        path = (a1, a2)
    
    else:
        path = (a2, a1)

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


def get_next_cut(a, vertices, same_ray=True, sign=1):
    n_vertices = vertices.shape[0]
    criterion = 0 if same_ray else 1

    ray_a = int(vertices[a, 2])

    shifted_indices = np.mod(a + 1 + np.arange(n_vertices), n_vertices)

    if sign != 0:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) >= 0,
                np.bitwise_and(
                    np.abs(vertices[shifted_indices, 2].astype(np.int32) - ray_a) == criterion,
                    vertices[shifted_indices, -2].astype(np.int32) == sign
                )
            )
        )[0]
    else:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) >= 0,
                np.abs(vertices[shifted_indices, 2].astype(np.int32) - ray_a) == criterion
            )
        )[0]

    if len(b) == 0:
        return None
    
    b = shifted_indices[b[0]]

    return b


def get_prev_cut(a, vertices, same_ray=True, sign=1):
    n_vertices = vertices.shape[0]
    criterion = 0 if same_ray else 1

    ray_a = int(vertices[a, 2])

    shifted_indices = np.mod(a + np.arange(n_vertices), n_vertices)
    if sign != 0:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) >= 0,
                np.bitwise_and(
                    np.abs(vertices[shifted_indices, 2].astype(np.int32) - ray_a) == criterion,
                    vertices[shifted_indices, -2].astype(np.int32) == sign
                )
            )
        )[0]
    else:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) >= 0,
                np.abs(vertices[shifted_indices, 2].astype(np.int32) - ray_a) == criterion
            )
        )[0]
    
    if len(b) == 0:
        return None

    b = shifted_indices[b[-1]]

    return b


def get_next_on_ray(a, vertices, sign=1):
    n_vertices = vertices.shape[0]
    ray_a = int(vertices[a, 2])
    next_pos = int(vertices[a, -1]) + 1

    shifted_indices = np.mod(a + 1 + np.arange(n_vertices), n_vertices)
    if sign != 0:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) == ray_a,
                np.bitwise_and(
                    vertices[shifted_indices, -1].astype(np.int32) == next_pos,
                    vertices[shifted_indices, -2].astype(np.int32) == sign
                )
            )
        )[0]
    else:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) == ray_a,
                vertices[shifted_indices, -1].astype(np.int32) == next_pos
            )
        )[0]
        
    if len(b) == 0:
        return None
    
    b = shifted_indices[b[0]]

    return b


def get_prev_on_ray(a, vertices, sign=1):
    n_vertices = vertices.shape[0]
    ray_a = int(vertices[a, 2])
    prev_pos = int(vertices[a, -1]) - 1

    shifted_indices = np.mod(a + np.arange(n_vertices), n_vertices)
    if sign != 0:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) == ray_a,
                np.bitwise_and(
                    vertices[shifted_indices, -1].astype(np.int32) == prev_pos,
                    vertices[shifted_indices, -2].astype(np.int32) == sign
                )
            )
        )[0]
    else:
        b = np.where(
            np.bitwise_and(
                vertices[shifted_indices, 2].astype(np.int32) == ray_a,
                vertices[shifted_indices, -1].astype(np.int32) == prev_pos
            )
        )[0]
    
    if len(b) == 0:
        return None

    b = shifted_indices[b[-1]]

    return b


def check_adjacency(a1, a2, vertices, left2right=True):
    # The cut point a2 is adjacent to the cut a1 to the left
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


def left_condition_1(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None
    
    is_left_valid = check_adjacency(a1, a2, vertices, left2right=True)

    return is_left_valid


def left_condition_2(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    b1 = get_next_cut(a1, vertices, same_ray=False, sign=1)
    b2 = get_prev_cut(a2, vertices, same_ray=False, sign=-1)

    if b1 is None or b2 is None:
        return None
    
    if not (check_adjacency(a1, b1, vertices, left2right=True) and check_adjacency(b2, a2, vertices, left2right=True)):
        return None

    child_id = get_alias_cut(b1, b2, vertices, left2right=True)
    validity_tree[child_id] = {}
    is_left_valid = check_left_validity(b1, b2, vertices, validity_tree=validity_tree[child_id], **kwargs)

    if is_left_valid == 'Explored':
        validity_tree[child_id] = 'Exploring'
        is_left_valid = None

    return is_left_valid


def left_condition_3(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    a1_pos = int(vertices[a1, -1])
    a2_pos = int(vertices[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if abs(a1_pos - a2_pos) - 1 < 2:
        return None

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None

    ray_a = int(vertices[a1, 2])
    is_valid = None
    for mc in filter(lambda mc: mc[1] == ray_a, kwargs['min_crest_cuts']):
        a3 = mc[2]
        a4 = mc[3]

        # Check that all intersection points are different
        if len(set({a1, a2, a3, a4})) < 4:
            continue

        # Check the order of the intersection vertices on the current ray
        if not (vertices[a1, -1] < vertices[a3, -1] < vertices[a4, -1] < vertices[a2, -1]):
            continue

        # Check if point a3' and a1 are left valid
        child_id = get_alias_cut(a1, a3, vertices, left2right=True)
        validity_tree[child_id] = {}
        is_left_valid = check_left_validity(a1, a3, vertices, validity_tree=validity_tree[child_id], **kwargs)
    
        if is_left_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_left_valid = None
            
        # Check if point a2' and a4 are left valid
        child_id = get_alias_cut(a4, a2, vertices, left2right=True)
        validity_tree[child_id] = {}
        is_right_valid = check_left_validity(a4, a2, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_right_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_right_valid = None

        if is_left_valid is None and is_right_valid is None:
            continue
        elif is_right_valid is None:
            pair_is_valid = is_left_valid
        elif is_left_valid is None:
            pair_is_valid = is_right_valid
        else:
            pair_is_valid = is_left_valid & is_right_valid

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid
        
    return is_valid


def left_condition_4(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None

    # Get the previous cut of a2'
    a3 = get_prev_cut(a2, vertices, same_ray=True, sign=1)
    if a3 is None:
        return None

    # Verify that a2' and a3 are maximal crest cuts
    if not any(filter(lambda mc: mc[2] == a3 and mc[3] == a2, kwargs['max_crest_cuts'])):
        return None

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    n_vertices_on_ray = np.max(vertices[vertices[:, 2].astype(np.int32) == int(vertices[a3, 2]), -1])
    while int(vertices[a_p, -1]) < n_vertices_on_ray:
        a_p = get_next_on_ray(a_p, vertices, sign=-1)
        if a_p is None:
            break

        # Check if that point and a1 are left valid
        child_id = get_alias_cut(a1, a_p, vertices, left2right=True)
        validity_tree[child_id] = {}
        is_left_valid = check_left_validity(a1, a_p, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_left_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_left_valid = None

        # Check if that point and a3 are right valid
        child_id = get_alias_cut(a3, a_p, vertices, left2right=False)
        validity_tree[child_id] = {}
        is_right_valid = check_right_validity(a3, a_p, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_right_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_right_valid = None

        if is_left_valid is None and is_right_valid is None:
            continue
        elif is_right_valid is None:
            pair_is_valid = is_left_valid
        elif is_left_valid is None:
            pair_is_valid = is_right_valid
        else:
            pair_is_valid = is_left_valid & is_right_valid

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid


def left_condition_5(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None

    # Get the next cut of a1
    a3 = get_next_cut(a1, vertices, same_ray=True, sign=-1)
    if a3 is None:
        return None
    
    # Verify that a1 and a3' are maximal crest cuts
    if not any(filter(lambda mc: mc[2] == a1 and mc[3] == a3, kwargs['max_crest_cuts'])):
        return None

    # Get all points to the left of a3 in the same ray
    a_p = a3
    is_valid = None
    
    while int(vertices[a_p, -1]) > 0:
        a_p = get_prev_on_ray(a_p, vertices, sign=1)
        if a_p is None:
            break

        # Check if that point and a2' are left valid
        child_id = get_alias_cut(a_p, a2, vertices, left2right=True)
        validity_tree[child_id] = {}
        is_left_valid = check_left_validity(a_p, a2, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_left_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_left_valid = None

        # Check if that point and a3 are right valid
        child_id = get_alias_cut(a_p, a3, vertices, left2right=False)
        validity_tree[child_id] = {}
        is_right_valid = check_right_validity(a_p, a3, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_right_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_right_valid = None

        if is_left_valid is None and is_right_valid is None:
            continue
        elif is_right_valid is None:
            pair_is_valid = is_left_valid
        elif is_left_valid is None:
            pair_is_valid = is_right_valid
        else:
            pair_is_valid = is_left_valid & is_right_valid

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid


def right_condition_1(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    is_right_valid = check_adjacency(a1, a2, vertices, left2right=False)

    return is_right_valid


def right_condition_2(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None
    
    b1 = get_prev_cut(a1, vertices, same_ray=False, sign=1)
    b2 = get_next_cut(a2, vertices, same_ray=False, sign=-1)

    if b1 is None or b2 is None:
        return None

    if not (check_adjacency(a1, b1, vertices, left2right=False) and check_adjacency(b2, a2, vertices, left2right=False)):
        return None

    child_id = get_alias_cut(b1, b2, vertices, left2right=False)
    validity_tree[child_id] = {}
    is_right_valid = check_right_validity(b1, b2, vertices, validity_tree=validity_tree[child_id], **kwargs)
    if is_right_valid == 'Explored':
        validity_tree[child_id] = 'Exploring'
        is_right_valid = None

    return is_right_valid


def right_condition_3(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    a1_pos = int(vertices[a1, -1])
    a2_pos = int(vertices[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if abs(a1_pos - a2_pos) - 1 < 2:
        return None

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None

    ray_a = int(vertices[a1, 2])
    is_valid = None
    
    for mc in filter(lambda mc: mc[1] == ray_a, kwargs['max_crest_cuts']):
        a3 = mc[3]
        a4 = mc[2]

        # Check that all intersection points are different
        if len(set({a1, a2, a3, a4})) < 4:
            continue

        # Check the order of the intersection vertices on the current ray
        if not (vertices[a1, -1] < vertices[a3, -1] < vertices[a4, -1] < vertices[a2, -1]):
            continue

        # Check if point a3' and a1 are right valid
        child_id = get_alias_cut(a1, a3, vertices, left2right=False)
        validity_tree[child_id] = {}
        is_left_valid = check_right_validity(a1, a3, vertices, validity_tree=validity_tree, **kwargs)
        if is_left_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_left_valid = None

        # Check if point a2' and a4 are right valid
        child_id = get_alias_cut(a4, a2, vertices, left2right=False)
        validity_tree[child_id] = {}
        is_right_valid = check_right_validity(a4, a2, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_right_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_right_valid = None

        if is_left_valid is None and is_right_valid is None:
            continue
        elif is_right_valid is None:
            pair_is_valid = is_left_valid
        elif is_left_valid is None:
            pair_is_valid = is_right_valid
        else:
            pair_is_valid = is_left_valid & is_right_valid

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid
        
    return is_valid


def right_condition_4(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None

    a3 = get_next_cut(a2, vertices, same_ray=True, sign=1)
    if a3 is None:
        return None

    # Verify that a2' and a3 are minimal crest cuts
    if not any(filter(lambda mc: mc[2] == a2 and mc[3] == a3, kwargs['min_crest_cuts'])):
        return None

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    n_vertices_on_ray = np.max(vertices[vertices[:, 2].astype(np.int32) == int(vertices[a3, 2]), -1])

    while int(vertices[a_p, -1]) < n_vertices_on_ray:
        a_p = get_next_on_ray(a_p, vertices, sign=-1)
        if a_p is None:
            break

        # Check if that point and a1 are right valid
        child_id = get_alias_cut(a1, a_p, vertices, left2right=False)
        validity_tree[child_id] = {}
        is_left_valid = check_right_validity(a1, a_p, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_left_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_left_valid = None

        # Check if that point and a3 are left valid
        child_id = get_alias_cut(a3, a_p, vertices, left2right=True)
        validity_tree[child_id] = {}
        is_right_valid = check_left_validity(a3, a_p, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_right_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_right_valid = None

        if is_left_valid is None and is_right_valid is None:
            continue
        elif is_right_valid is None:
            pair_is_valid = is_left_valid
        elif is_left_valid is None:
            pair_is_valid = is_right_valid
        else:
            pair_is_valid = is_left_valid & is_right_valid

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid
            
    return is_valid


def right_condition_5(a1, a2, vertices, validity_tree, **kwargs):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices[:, 2].astype(np.int32) == int(vertices[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None

    a3 = get_prev_cut(a1, vertices, same_ray=True, sign=-1)
    if a3 is None:
        return None
    
    # Verify that a1 and a3' are minimal crest cuts
    if not any(filter(lambda mc: mc[2] == a3 and mc[3] == a1, kwargs['min_crest_cuts'])):
        return None

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    
    while int(vertices[a_p, -1]) > 0:
        a_p = get_prev_on_ray(a_p, vertices, sign=1)
        if a_p is None:
            break

        # Check if that point and a1 are right valid
        child_id = get_alias_cut(a_p, a2, vertices, left2right=False)
        validity_tree[child_id] = {}
        is_right_valid = check_right_validity(a_p, a2, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_right_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_right_valid = None

        # Check if that point and a3 are left valid
        child_id = get_alias_cut(a_p, a3, vertices, left2right=True)
        validity_tree[child_id] = {}
        is_left_valid = check_left_validity(a_p, a3, vertices, validity_tree=validity_tree[child_id], **kwargs)
        if is_left_valid == 'Explored':
            validity_tree[child_id] = 'Exploring'
            is_left_valid = None

        if is_left_valid is None and is_right_valid is None:
            continue
        elif is_right_valid is None:
            pair_is_valid = is_left_valid
        elif is_left_valid is None:
            pair_is_valid = is_right_valid
        else:
            pair_is_valid = is_left_valid & is_right_valid

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid


def left_invalid_condition_1(a1, a2, vertices):
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

    a3 = get_next_cut(a1, vertices, same_ray=True, sign=1)
    if a3 is None:
        return None
    
    if not check_adjacency(a1, a3, vertices, left2right=True):
        return None

    return not (vertices[a1, -1] < vertices[a3, -1] < vertices[a2, -1])


def left_invalid_condition_2(a1, a2, vertices):
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

    a3 = get_prev_cut(a2, vertices, same_ray=True, sign=1)
    if a3 is None:
        return None
    
    if not check_adjacency(a3, a2, vertices, left2right=True):
        return None
    
    return not (vertices[a1, -1] < vertices[a3, -1] < vertices[a2, -1])


def left_invalid_condition_3(a1, a2, vertices):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None

    b1 = get_prev_cut(a2, vertices, same_ray=False, sign=0)
    b2 = get_next_cut(a1, vertices, same_ray=False, sign=0)

    if b1 is None or b2 is None:
        return None

    if not (check_adjacency(a1, b2, vertices, left2right=True) and check_adjacency(b1, a2, vertices, left2right=True)):
        return None

    # Get all intersections between b2 and a1, and between a2 and b1
    n_vertices = vertices.shape[0]
    shifted_indices = np.mod(a1 + 1 + np.arange(n_vertices), n_vertices)
    r = (b2 - a1 - 1) + (0 if b2 > a1 else n_vertices)
    a1_b2_int = list(np.where(vertices[shifted_indices[:r], 2].astype(np.int32) < -1)[0])
    a1_b2_int = set(shifted_indices[a1_b2_int])

    shifted_indices = np.mod(b1 + 1 + np.arange(n_vertices), n_vertices)
    r = (a2 - b1 - 1) + (0 if a2 > b1 else n_vertices)
    b1_a2_int = list(np.where(vertices[shifted_indices[:r], 2].astype(np.int32) < -1)[0])
    b1_a2_int = set(shifted_indices[b1_a2_int])

    # This condition does not apply if only one segment contains any intersection
    if len(a1_b2_int) == 0 or len(b1_a2_int) == 0:
        return None

    # If there is at least one intersection point in both segments, this cut is right invalid
    return not len(a1_b2_int.intersection(b1_a2_int)) > 0


def right_invalid_condition_1(a1, a2, vertices):
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

    a3 = get_prev_cut(a1, vertices, same_ray=True, sign=-1)
    if a3 is None:
        return None
    
    if not check_adjacency(a3, a1, vertices, left2right=False):
        return None

    return not (vertices[a1, -1] < vertices[a3, -1] < vertices[a2, -1])


def right_invalid_condition_2(a1, a2, vertices):
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

    a3 = get_next_cut(a2, vertices, same_ray=True, sign=1)
    if a3 is None:
        return None

    if not check_adjacency(a2, a3, vertices, left2right=False):
        return None

    return not (vertices[a1, -1] < vertices[a3, -1] < vertices[a2, -1])


def right_invalid_condition_3(a1, a2, vertices):
    if vertices[a1, 3] < 0.5 or vertices[a2, 3] > 0.5:
        return None
    
    b1 = get_next_cut(a2, vertices, same_ray=False, sign=0)
    b2 = get_prev_cut(a1, vertices, same_ray=False, sign=0)

    if b1 is None or b2 is None:
        return None

    if not (check_adjacency(b1, a2, vertices, left2right=False) and check_adjacency(a1, b2, vertices, left2right=False)):
        return None

    # Get all intersections between b2 and a1, and between a2 and b1
    n_vertices = vertices.shape[0]
    shifted_indices = np.mod(b2 + 1 + np.arange(n_vertices), n_vertices)
    r = (a1 - b2 - 1) + (0 if a1 > b2 else n_vertices)
    b2_a1_int = list(np.where(vertices[shifted_indices[:r], 2].astype(np.int32) < -1)[0])
    b2_a1_int = set(shifted_indices[b2_a1_int])

    shifted_indices = np.mod(a2 + 1 + np.arange(n_vertices), n_vertices)
    r = (b1 - a2 - 1) + (0 if b1 > a2 else n_vertices)
    a2_b1_int = list(np.where(vertices[shifted_indices[:r], 2].astype(np.int32) < -1)[0])
    a2_b1_int = set(shifted_indices[a2_b1_int])

    # This condition does not apply if only one segment contains any intersection
    if len(b2_a1_int) == 0 or len(a2_b1_int) == 0:
        return None

    # If there is at least one intersection point in both segments, this cut is right invalid
    return not len(b2_a1_int.intersection(a2_b1_int)) > 0


all_left_valid_conditions = [left_condition_1, left_condition_2, left_condition_3, left_condition_4, left_condition_5]
all_right_valid_conditions = [right_condition_1, right_condition_2, right_condition_3, right_condition_4, right_condition_5]

all_left_invalid_conditions = [left_invalid_condition_1, left_invalid_condition_2, left_invalid_condition_3]
all_right_invalid_conditions = [right_invalid_condition_1, right_invalid_condition_2, right_invalid_condition_3]


def check_left_validity(a1, a2, vertices, max_crest_cuts, min_crest_cuts, validity_tree, visited):
    cut_id = get_alias_cut(a1, a2, vertices, plot_it=False, left2right=True)
    left_is_valid, _ = visited.get(cut_id, (None, {}))

    if left_is_valid is None:        
        visited[cut_id] = ('Explored', {})

        left_is_valid = True
        for condition in all_left_invalid_conditions:
            resp = condition(a1, a2, vertices)
            if resp is not None:
                left_is_valid &= resp
            if not left_is_valid:
                break    

        if left_is_valid:
            # This cut is not left valid until the contrary is proven
            left_is_valid = False
            for condition in all_left_valid_conditions:
                resp = condition(a1, a2, vertices, max_crest_cuts=max_crest_cuts, min_crest_cuts=min_crest_cuts, validity_tree=validity_tree, visited=visited)
                if resp is not None:
                    left_is_valid |= resp
            
        visited[cut_id] = (left_is_valid, {})

    return left_is_valid


def check_right_validity(a1, a2, vertices, max_crest_cuts, min_crest_cuts, validity_tree, visited):    
    cut_id = get_alias_cut(a1, a2, vertices, plot_it=False, left2right=False)
    right_is_valid, explored_path = visited.get(cut_id, (None, {}))
    
    if right_is_valid is None:
        visited[cut_id] = ('Explored', {})

        right_is_valid = True
        for condition in all_right_invalid_conditions:
            resp = condition(a1, a2, vertices)
            if resp is not None:
                right_is_valid &= resp
            if not right_is_valid:
                break    

        if right_is_valid:
            # This cut is not right valid until the contrary is proven
            right_is_valid = False
            for condition in all_right_valid_conditions:
                resp = condition(a1, a2, vertices, max_crest_cuts=max_crest_cuts, min_crest_cuts=min_crest_cuts, validity_tree=validity_tree, visited=visited)
                
                if resp is not None:
                    right_is_valid |= resp

        visited[cut_id] = (right_is_valid, validity_tree)
    
    else:
        # This node has been explored previously
        validity_tree.update(explored_path)

    return right_is_valid


def check_validity(a1, a2, vertices, max_crest_cuts, min_crest_cuts):
    cut_id = get_alias_cut(a1, a2, vertices)
    visited = {}

    validity_tree = {cut_id: {}}
    check_left_validity(a1, a2, vertices, max_crest_cuts, min_crest_cuts, validity_tree[cut_id], visited)
    check_right_validity(a1, a2, vertices, max_crest_cuts, min_crest_cuts, validity_tree[cut_id], visited)
    
    return visited, validity_tree


def traverse_tree(validity_tree, visited):
    valid_path = False
    child_keys = list(validity_tree.keys())

    for child_id in child_keys:
        if isinstance(validity_tree[child_id], dict) and validity_tree[child_id].keys():
            valid_subtree = traverse_tree(validity_tree[child_id], visited)
            
        else:
            # The child node is a leaf node in the tree
            if not isinstance(validity_tree[child_id], dict):
                print('Explored path ...', child_id)
            valid_subtree, _ = visited[child_id]

        valid_path |= valid_subtree
        if not valid_subtree:
            del validity_tree[child_id]
    
    return valid_path
