import numpy as np
import math
from functools import reduce
import matplotlib.pyplot as plt


def _get_shifted_indices(a1, a2, n_vertices, exclude_start=True, exclude_end=True):
    """ Shifts the position of the indices in a way that wraps them connecting the last index with the first one.

    Parameters
    ----------
    a1 : int
        The first index after wrapping and shifting the indices.
    a2 : int
        The first index after wrapping and shifting the indices.
    n_vertices : int
        The number of vertices in the polygon.
    exclude_start : bool
        If the first index (a1) will be excluded from the selected sequence of indices.
    exclude_end : bool
        If the last index (a2) will be excluded from the selected sequence of indices.

    Returns
    -------
    shifted_indices : numpy.ndarray
        The sequence of wrapped and shifted indices between a1 and a2.
    
    """
    shifted_indices = np.mod(a1 + np.arange(n_vertices), n_vertices)
    r = a2 - a1 + 1 + (0 if a2 > a1 else n_vertices)
    return shifted_indices[exclude_start:r - exclude_end]


def _get_cut_alias(a1, a2, vertices_info, left2right=True):
    ray_1 = int(vertices_info[a1, 2])
    sign_1 = '\'' if vertices_info[a1, 3] < 0.5 else ''
    ord_ray_1 = int(vertices_info[a1, 4])
    ray_2 = int(vertices_info[a2, 2])
    sign_2 = '\'' if vertices_info[a2, 3] < 0.5 else ''
    ord_ray_2 = int(vertices_info[a2, 4])

    cut_alias = '%s%i%s,%s%i%s,' % (chr(97 + ray_1), ord_ray_1+1, sign_1, chr(97 + ray_2), ord_ray_2+1, sign_2) + ('L' if left2right else 'R')
    return cut_alias


def _get_cut_id(a1, a2, left2right=True):
    """ Generate an identifier key for the cut between a1 and a2.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    left2right : bool, optional
        The direction of the condition tested for the validity of this cut.
        The default direction is from left to right on the polygon vertices.
    
    Returns
    -------
    cut_id : tuple
        A tuple containing the cut indices a1 and a2, 
        and the direction of the vertices of the polygon between indices a1 and a2.
    """
    cut_id = (a1, a2, 'L' if left2right else 'R')
    return cut_id


def _get_cut(a, vertices_info, next_cut=True, same_ray=True, sign=0):
    """ Looks for the next/previous valid vertex that can be used as cut along with vertex `a`.

    Parameters
    ----------
    a : int
        An index of a vertex on the polygon used as reference to look for a cut that satisfyes the given conditions.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    next_cut : bool, optional
        If the cut is looked for after index `a` from left to right on the polygon.
        Use False to look for the previous cut instead.
    same_ray : bool, optional
        If the next/previous cut index is looked for on the same ray to which `a` belongs.
    sign : int, optional
        The orientation (positive 1 /negative -1 / ambiguous 0) of the vertex index
        according to the ray that generated it.
    
    Returns
    -------
    cut_id : int
        The next/previous vertex index that satisfies the specifications given.
    """
    
    n_vertices = vertices_info.shape[0]
    ray_a = int(vertices_info[a, 2])
    
    if same_ray:
        criterion_0 = np.ones(n_vertices, dtype=np.bool)
        criterion_1 = vertices_info[:, 2].astype(np.int32) == ray_a

        shifted_indices = np.argsort(vertices_info[:, -1])
        pos_shift = np.where(vertices_info[shifted_indices, -1].astype(np.int32) == int(vertices_info[a, -1]))[0][-1 * next_cut]
        shifted_indices = shifted_indices[np.mod(pos_shift + next_cut + np.arange(n_vertices), n_vertices)]
    else:
        shifted_indices = np.mod(a + (1 if next_cut else 0) + np.arange(n_vertices), n_vertices)
        criterion_0 = vertices_info[:, 2].astype(np.int32) >= 0
        criterion_1 = vertices_info[:, 2].astype(np.int32) != ray_a
    
    if sign:
        criterion_2 = vertices_info[:, -2].astype(np.int32) == sign
    else:
        criterion_2 = np.ones(n_vertices, dtype=np.bool)
    
    criteria = criterion_0 * criterion_1 * criterion_2
    cut_id = np.where(criteria[shifted_indices])[0]

    if len(cut_id) == 0:
        return None
    
    cut_id = shifted_indices[cut_id[0 if next_cut else -1]]

    return cut_id


def _check_adjacency(a1, a2, vertices_info, left2right=True):
    """ Check if the cut point a2 is adjacent to the cut a1 to the left/right.
    
    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    left2right : bool, optional
        The direction of the condition tested for the validity of this cut.
        The default direction is from left to right on the polygon vertices_info
    
    Returns
    -------
    is_adjacent : bool
        If the indices a1 and a2 are adjacent when looked from left to right, or right to left.
    """
    n_vertices = vertices_info.shape[0]
    
    if left2right:
        shifted_indices = _get_shifted_indices(a1, a2, n_vertices, exclude_start=True, exclude_end=True)
    else:
        shifted_indices = _get_shifted_indices(a2, a1, n_vertices, exclude_start=True, exclude_end=True)

    is_adjacent = not np.any(vertices_info[shifted_indices, 2].astype(np.int32) >= 0)
    
    return is_adjacent


def _condition_1(a1, a2, vertices_info, check_left=True, **kwargs):
    """ First condition that a cut has to satisfy in order to be left/right valid.
        
    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right, or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut validity has already been expanded.
    children_ids : list 
        A list of tuples containing the indices of the explored cuts on which this cut depends on,
        and an identifier of ths condition number and direction.
    """
    children_ids = []
    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None, children_ids

    is_valid = _check_adjacency(a1, a2, vertices_info, left2right=check_left)
    if is_valid:
        children_ids.append([(-1, -1, 'self', '1%s' % ('L' if check_left else 'R'))])
    return is_valid, children_ids


def _condition_2(a1, a2, vertices_info, check_left=True, **kwargs):
    """ Second condition that a cut has to satisfy in order to be left/right valid.
        
    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right, or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut validity has already been expanded.
    children_ids : list 
        A list of tuples containing the indices of the explored cuts on which this cut depends on,
        and an identifier of ths condition number and direction.
    """
    children_ids = []
    is_valid = None

    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None, children_ids
    
    b1 = _get_cut(a1, vertices_info, next_cut=check_left, same_ray=False, sign=1)
    b2 = _get_cut(a2, vertices_info, next_cut=not check_left, same_ray=False, sign=-1)

    if b1 is None or b2 is None or not (_check_adjacency(a1, b1, vertices_info, left2right=check_left) and _check_adjacency(b2, a2, vertices_info, left2right=check_left)):
        return None, children_ids
    
    ray_b1 = int(vertices_info[b1, 2])
    ray_b2 = int(vertices_info[b2, 2])
    
    if ray_b1 != ray_b2:
        return None, children_ids

    is_valid, child_id = _check_validity(b1, b2, vertices_info, check_left=check_left, **kwargs)
    child_id = (*child_id, '2%s' % ('L' if check_left else 'R'))

    children_ids.append([child_id])
    is_valid = None if isinstance(is_valid, str) and is_valid == 'Explored' else is_valid
    
    return is_valid, children_ids


def _condition_3(a1, a2, vertices_info, check_left=True, **kwargs):
    """ Third condition that a cut has to satisfy in order to be left/right valid.
        
    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right, or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut validity has already been expanded.
    children_ids : list 
        A list of tuples containing the indices of the explored cuts on which this cut depends on,
        and an identifier of ths condition number and direction.
    """
    children_ids = []

    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None, children_ids

    a1_pos = int(vertices_info[a1, -1])
    a2_pos = int(vertices_info[a2, -1])

    # If there are no vertices_info between a1 and 2, continue with other condition
    if abs(a1_pos - a2_pos) - 1 < 2:
        return None, children_ids

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices_info[:, 2].astype(np.int32) == int(vertices_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    ray_a = int(vertices_info[a1, 2])
    is_valid = None

    for mc in filter(lambda mc: mc[1] == ray_a, kwargs['min_crest_cuts' if check_left else 'max_crest_cuts']):
        a3 = mc[2 if check_left else 3]
        a4 = mc[3 if check_left else 2]

        # Check that all intersection points are different
        if len(set({a1, a2, a3, a4})) < 4:
            continue

        # Check the order of the intersection vertices_info on the current ray
        if not (vertices_info[a1, -1] < vertices_info[a3, -1] < vertices_info[a4, -1] < vertices_info[a2, -1]):
            continue

        # Check if point a3' and a1 are left/right valid
        is_valid_1, child_id_1 = _check_validity(a1, a3, vertices_info, check_left=check_left, **kwargs)
        child_id_1 = (*child_id_1, '3%s' % ('L' if check_left else 'R'))

        # Check if point a2' and a4 are left/right valid
        is_valid_2, child_id_2 = _check_validity(a4, a2, vertices_info, check_left=check_left, **kwargs)
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


def _condition_4(a1, a2, vertices_info, check_left=True, **kwargs):
    """ Fourth condition that a cut has to satisfy in order to be left/right valid.
        
    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right, or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut validity has already been expanded.
    children_ids : list 
        A list of tuples containing the indices of the explored cuts on which this cut depends on,
        and an identifier of ths condition number and direction.
    """
    children_ids = []

    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None, children_ids

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices_info[:, 2].astype(np.int32) == int(vertices_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids
    
    a3 = _get_cut(a2, vertices_info, next_cut=True, same_ray=True, sign=1)
    if a3 is None:
        return None, children_ids

    # Verify that a2' and a3 are minimal crest cuts
    if not any(filter(lambda mc: mc[3 if check_left else 2] == a2 and mc[2 if check_left else 3] == a3, kwargs['max_crest_cuts' if check_left else 'min_crest_cuts'])):
        return None, children_ids

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    n_vertices_on_ray = np.max(vertices_info[vertices_info[:, 2].astype(np.int32) == int(vertices_info[a3, 2]), -1])
    set_id = 0

    while int(vertices_info[a_p, -1]) < n_vertices_on_ray:
        set_id += 1

        a_p = _get_cut(a_p, vertices_info, next_cut=True, same_ray=True, sign=-1)
        if a_p is None:
            break

        # Check if that point and a1 are left/right valid
        is_valid_1, child_id_1 = _check_validity(a1, a_p, vertices_info, check_left=check_left, **kwargs)
        child_id_1 = (*child_id_1, '4%s' % ('L' if check_left else 'R'))

        # Check if that point and a3 are right/left valid
        is_valid_2, child_id_2 = _check_validity(a3, a_p, vertices_info, check_left=not check_left, **kwargs)
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


def _condition_5(a1, a2, vertices_info, check_left=True, **kwargs):
    """ Fifth condition that a cut has to satisfy in order to be left/right valid.
        
    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right, or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut validity has already been expanded.
    children_ids : list 
        A list of tuples containing the indices of the explored cuts on which this cut depends on,
        and an identifier of ths condition number and direction.
    """
    children_ids = []

    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None, children_ids

    # This condition is applied only if there are four or more cut points in the same ray
    ver_in_ray_ids = list(np.where(vertices_info[:, 2].astype(np.int32) == int(vertices_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    a3 = _get_cut(a1, vertices_info, next_cut=False, same_ray=True, sign=-1)
    if a3 is None:
        return None, children_ids
    
    # Verify that a1 and a3' are minimal crest cuts
    if not any(filter(lambda mc: mc[3 if check_left else 2] == a3 and mc[2 if check_left else 3] == a1, kwargs['max_crest_cuts' if check_left else 'min_crest_cuts'])):
        return None, children_ids

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    set_id = 0
    while int(vertices_info[a_p, -1]) > 0:
        set_id += 1

        a_p = _get_cut(a_p, vertices_info, next_cut=False, same_ray=True, sign=1)
        if a_p is None:
            break

        # Check if that point and a1 are right valid
        is_valid_1, child_id_1 = _check_validity(a_p, a2, vertices_info, check_left=check_left, **kwargs)
        child_id_1 = (*child_id_1, '5%s' % ('L' if check_left else 'R'))

        # Check if that point and a3 are left valid
        is_valid_2, child_id_2 = _check_validity(a_p, a3, vertices_info, check_left=not check_left, **kwargs)
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


def _invalidity_condition_1(a1, a2, vertices_info):
    """ First condition that can turn a cut to be left invalid.

    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 has no invalidity according with this condition.
        This returns None when this condition can not be tested.
    """
    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None
    
    a1_pos = int(vertices_info[a1, -1])
    a2_pos = int(vertices_info[a2, -1])

    # If there are no vertices_info between a1 and 2, continue with other condition
    if a1_pos - a2_pos != 2:
        return None

    ver_in_ray_ids = list(np.where(vertices_info[:, 2].astype(np.int32) == int(vertices_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 3:
        return None

    a3 = _get_cut(a1, vertices_info, next_cut=True, same_ray=True, sign=-1)
    
    if a3 is None:
        return None
    
    if not _check_adjacency(a3, a1, vertices_info, left2right=True):
        return None

    is_valid = not (vertices_info[a1, -1] < vertices_info[a3, -1] < vertices_info[a2, -1])

    return is_valid


def _invalidity_condition_2(a1, a2, vertices_info):
    """ Second condition that can turn a cut to be left invalid.

    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 has no invalidity according with this condition.
        This returns None when this condition can not be tested.
    """
    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None
    
    a1_pos = int(vertices_info[a1, -1])
    a2_pos = int(vertices_info[a2, -1])

    # If there are no vertices_info between a1 and 2, continue with other condition
    if a1_pos - a2_pos != 2:
        return None

    ver_in_ray_ids = list(np.where(vertices_info[:, 2].astype(np.int32) == int(vertices_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 3:
        return None

    a3 = _get_cut(a2, vertices_info, next_cut=False, same_ray=True, sign=-1)
    
    if a3 is None:
        return None

    if not _check_adjacency(a2, a3, vertices_info, left2right=True):
        return None

    is_valid = not (vertices_info[a1, -1] < vertices_info[a3, -1] < vertices_info[a2, -1])
    return is_valid


def _invalidity_condition_3(a1, a2, vertices_info, tolerance=1e-4):
    """ Third condition that can turn a cut to be left invalid.

    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 has no invalidity according with this condition.
        This returns None when this condition can not be tested.
    """
    if vertices_info[a1, 3] < 0.5 or vertices_info[a2, 3] > 0.5:
        return None
    
    b2 = _get_cut(a1, vertices_info, next_cut=True, same_ray=False, sign=0)
    b1 = _get_cut(a2, vertices_info, next_cut=False, same_ray=False, sign=0)

    if b1 is None or b2 is None:
        return None

    if not (_check_adjacency(b1, a2, vertices_info, left2right=True) and _check_adjacency(a1, b2, vertices_info, left2right=True)):
        return None

    n_vertices = vertices_info.shape[0]
    # Get all intersections between b2 and a1, and between a2 and b1
    shifted_indices_1 = _get_shifted_indices(a1, b2, n_vertices, exclude_start=True, exclude_end=True)
    shifted_indices_2 = _get_shifted_indices(b1, a2, n_vertices, exclude_start=True, exclude_end=True)
    
    b2_a1_int = list(np.where(vertices_info[shifted_indices_1, 2].astype(np.int32) < -1)[0])
    a2_b1_int = list(np.where(vertices_info[shifted_indices_2, 2].astype(np.int32) < -1)[0])

    # This condition does not apply if only one segment contains any intersection
    if len(b2_a1_int) == 0 or len(a2_b1_int) == 0:
        return None

    int_1 = vertices_info[shifted_indices_1[b2_a1_int], :2]
    int_1 = int_1 / np.linalg.norm(int_1, axis=1)[..., np.newaxis]
    int_2 = vertices_info[shifted_indices_2[a2_b1_int], :2]
    int_2 = int_2 / np.linalg.norm(int_2, axis=1)[..., np.newaxis]

    # If there is at least one intersection point in both segments, this cut is right invalid
    is_valid = not (np.matmul(int_1, int_2.transpose()) >= 1 - tolerance).any()
    return is_valid


all_valid_conditions = [_condition_5, _condition_4, _condition_3, _condition_2, _condition_1]
all_invalid_conditions = [_invalidity_condition_1, _invalidity_condition_2, _invalidity_condition_3]


def _check_validity(a1, a2, vertices_info, max_crest_cuts, min_crest_cuts, visited, check_left=True):
    """ Checks recursively the validity of the cut between indices a1 and a2 of the polygon.

    Parameters 
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vertices_info : numpy.ndarray
        The array containing the information about the polygon and the characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right, or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is valid or is being explored
        This returns None when this cuts dependencies have been already explored before.
    cut_id : tuple
        A tuple containing the indices a1 and a2, 
        and the direction of the vertices of the polygon between indices a1 and a2 of the current cut.

    """
    cut_id = _get_cut_id(a1, a2, left2right=check_left)

    # Check if this node has been visited before
    is_valid, children_ids = visited.get(cut_id, [None, []])
    
    if is_valid is None:
        visited[cut_id] = ['Explored', children_ids]
        is_valid = True
        
        if check_left:
            for condition in all_invalid_conditions:
                resp = condition(a1, a2, vertices_info)
                if resp is not None:
                    is_valid &= resp
                if not is_valid:
                    break

        if is_valid:
            # This cut is not left valid until the contrary is proven
            is_valid = None
            for condition in all_valid_conditions:
                resp, cond_children_ids = condition(a1, a2, vertices_info, 
                        max_crest_cuts=max_crest_cuts, 
                        min_crest_cuts=min_crest_cuts, 
                        visited=visited, 
                        check_left=check_left)
                if resp is not None:
                    children_ids += cond_children_ids
                    if is_valid is None:
                        is_valid = resp
                    else:
                        is_valid |= resp
        
        visited[cut_id] = [is_valid, children_ids]

    return is_valid, cut_id


def _traverse_tree(root, visited, path=None):
    """ Traverse the tree of validity of the cuts that were tested previously with `_check_validity`.
    Parameters 
    ----------
    root : tuple
        A tuple containing the indices a1 and a2, 
        and the direction of the vertices of the polygon between indices a1 and a2 of the current cut used as root.
    visited : dict
        A dictionary containing the dependencies and validities of the already visited cuts.
    path : list or None
        A list with the cut identifiers of the already visited cuts. This prevents infinite recursion on cyclic graphs.
    
    Returns
    -------
    validity : bool
        The validity of the current branch being traversed.
    """
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
        sibling_validity = all(map(lambda sib_path: _traverse_tree(sib_path, visited, list(path) + [root]), sib_cond))
        
        if not sibling_validity:
            visited[root][1].remove(sib_cond)
        
        validity |= sibling_validity
    
    visited[root][0] = validity
    return validity


def _immerse_valid_tree(root, visited, vertices_info, polys_idx):
    """ Traverses the valid paths according to the visited dictionary to get the set of non-overlapping sub polygons.
        This traverses only one of the possible valid immersions for simplicity.

    Parameters 
    ----------
    root : tuple
        A tuple containing the indices a1 and a2, 
        and the direction of the vertices_info of the polygon between indices a1 and a2 of the current cut used as root.
    visited : dict
        A dictionary containing the dependencies and validities of the already visited cuts.
    n_vertices : int
        The number of vertices_info in the polygon.
    polys_idx : list
        A list of the indices of the polygons that are being discovered.

    Returns
    -------
    immersion : dict
        The children sub-tree of valid conditions.
    sub_poly : list
        A list of indices of all sub-polygons generated form children conditions.
    """
    n_vertices = vertices_info.shape[0]
    root = root if len(root) == 3 else root[:-1]
    immersion = {root:{}}
    _, conditions = visited.get(root, (False, None))

    a1, a2 = root[:2]
    left2right = root[2][-1] == 'L'
    sub_poly = []

    if conditions[0][0][0] < 0:
        # The base case is reached when a cut that is left/right valid by condition 1.
        if left2right:
            sub_poly.append(_get_shifted_indices(a1, a2, n_vertices, exclude_start=False, exclude_end=False))
        else:
            sub_poly.append(_get_shifted_indices(a2, a1, n_vertices, exclude_start=False, exclude_end=False))

        return immersion, sub_poly

    # Mark all children conditions as visited. This prevents children conditions from re-visiting cuts that belong to the parent cut.
    last_child_cond_id = None
    for sib_cond in conditions:
        for child_1 in sib_cond:
            child_owner, child_conditions = visited.get(child_1[:-1], (None, None))
            if child_owner is None and child_conditions is not None:
                visited[child_1[:-1]][0] = root

        child_1 = sib_cond[0]
        child_cond_id = child_1[-1][0]

        cut_owner, _ = visited.get(child_1[:-1], (None, None))
        if cut_owner is not None and cut_owner != root:
            continue
        # if last_child_cond_id is not None and last_child_cond_id != child_cond_id:
        #     break

        last_child_cond_id = child_cond_id

        if len(sib_cond) == 1:
            child_left2right = child_1[-1][1] == 'L'

            sub_immersion, child_poly = _immerse_valid_tree(child_1, visited, vertices_info, polys_idx)
            immersion[root].update(sub_immersion)

            # If this cut was selected by complying with condition 2, add the child_1 path, and send it to the parent path
            b1, b2 = child_1[:2]
            if child_cond_id == '2':
                if child_left2right:
                    exclude_child_end = False if len(child_poly) == 0 else child_poly[0][0] in [a1, b1]
                    exclude_child_start = False if len(child_poly) == 0 else child_poly[-1][-1] in [a2, b2]

                    sub_poly.append(_get_shifted_indices(a1, b1, n_vertices, exclude_start=False, exclude_end=exclude_child_end))
                    sub_poly += child_poly
                    sub_poly.append(_get_shifted_indices(b2, a2, n_vertices, exclude_start=exclude_child_start, exclude_end=False))
                else:
                    exclude_child_end = False if len(child_poly) == 0 else child_poly[0][0] in [a2, b2]
                    exclude_child_start = False if len(child_poly) == 0 else child_poly[-1][-1] in [a1, b1]

                    sub_poly.append(_get_shifted_indices(a2, b2, n_vertices, exclude_start=False, exclude_end=exclude_child_end))
                    sub_poly += child_poly
                    sub_poly.append(_get_shifted_indices(b1, a1, n_vertices, exclude_start=exclude_child_start, exclude_end=False))
            else:
                sub_poly = child_poly

        else:
            # Children cuts that where generated using rules 3, 4, and 5 have a special way to be merged with their respective parents.
            child_1, child_2 = sib_cond
            child_left2right = child_1[-1][1] == 'L'

            if child_cond_id == '3':
                sub_immersion, child_poly = _immerse_valid_tree(child_1, visited, vertices_info, polys_idx)
                if len(child_poly):
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_immersion, child_poly = _immerse_valid_tree(child_2, visited, vertices_info, polys_idx)
                if len(child_poly):
                    immersion[root].update(sub_immersion)                
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                if child_left2right:
                    sub_poly.append(_get_shifted_indices(child_1[1], child_2[0], n_vertices, exclude_start=False, exclude_end=False))
                else:
                    sub_poly.append(_get_shifted_indices(child_2[0], child_1[1], n_vertices, exclude_start=False, exclude_end=False))

            elif child_cond_id == '4':
                sub_immersion, child_poly = _immerse_valid_tree(child_1, visited, vertices_info, polys_idx)
                immersion[root].update(sub_immersion)
                sub_poly += child_poly

                if child_left2right:
                    sub_poly.append(_get_shifted_indices(child_2[0], a2, n_vertices, exclude_start=False, exclude_end=False))
                else:
                    sub_poly.append(_get_shifted_indices(a2, child_2[0], n_vertices, exclude_start=False, exclude_end=False))

                polys_idx.append(np.concatenate(sub_poly, axis=0))

                sub_immersion, child_poly = _immerse_valid_tree(child_2, visited, vertices_info, polys_idx)
                if len(child_poly):
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_poly = []

            elif child_cond_id == '5':
                sub_immersion, child_poly = _immerse_valid_tree(child_1, visited, vertices_info, polys_idx)
                immersion[root].update(sub_immersion)
                sub_poly += child_poly

                if child_left2right:
                    sub_poly.append(_get_shifted_indices(a1, child_2[1], n_vertices, exclude_start=False, exclude_end=False))
                else:
                    sub_poly.append(_get_shifted_indices(child_2[1], a1, n_vertices, exclude_start=False, exclude_end=False))

                polys_idx.append(np.concatenate(sub_poly, axis=0))

                sub_immersion, child_poly = _immerse_valid_tree(child_2, visited, vertices_info, polys_idx)
                if len(child_poly):
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_poly = []
        # break

    return immersion, sub_poly


def _get_root_indices(vertices_info):
    """ Finds the indices of the vertices that define the root cut used to recurse the polygon subdivision algorithm.
    
    Parameters 
    ----------
    vertices_info : numpy.ndarray
        An array containing the coordinates of the polygon vertices and additional information about each.
    
    Returns
    -------
    left_idx : int
        The positional index of the root left vertex
    right_idx : int
        The positional index of the root right vertex

    """
    n_vertices = vertices_info.shape[0]

    # Non-intersection points:
    org_ids = np.where(vertices_info[:, 2].astype(np.int32) == -1)[0]

    root_vertex = org_ids[np.argmax(vertices_info[org_ids, 1])]

    shifted_indices = np.mod(root_vertex + 1 + np.arange(n_vertices), n_vertices)
    right_idx = np.where(vertices_info[shifted_indices, 2].astype(np.int32) == 0)[0][0]
    right_idx = shifted_indices[right_idx]

    shifted_indices = np.mod(root_vertex + np.arange(n_vertices), n_vertices)
    left_idx = np.where(vertices_info[shifted_indices, 2].astype(np.int32) == 0)[0][-1]
    left_idx = shifted_indices[left_idx]

    return left_idx, right_idx


def _get_crest_ids(vertices, tolerance=1e-4):
    """ Finds the positional indices of the crest points.
    Only crests where there is a left turn on the polygon perimeter are considered.
    
    Parameters 
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    tolerance : float, optional
        A tolerance to determine if two vertices are at `almost` the same height.
    
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

    # Get the direction of the polygon when walking its perimeter.
    # Only crest points of left turns are considered
    diff_x_prev = vertices[1:, 0] - vertices[:-1, 0]
    diff_x_next = -diff_x_prev
    diff_x_prev = np.insert(diff_x_prev, 0, vertices[0, 0] - vertices[-1, 0])
    diff_x_next = np.append(diff_x_next, vertices[-1, 0] - vertices[0, 0])
    direction = np.bitwise_and(diff_x_prev >= -tolerance, diff_x_next <= tolerance)

    diff_y_prev = vertices[1:, 1] - vertices[:-1, 1]
    diff_y_next = -diff_y_prev

    diff_y_prev = np.insert(diff_y_prev, 0, vertices[0, 1] - vertices[-1, 1])
    diff_y_next = np.append(diff_y_next, vertices[-1, 1] - vertices[0, 1])

    higher_than_prev = diff_y_prev > tolerance
    higher_than_next = diff_y_next > tolerance

    lower_than_prev = diff_y_prev < -tolerance
    lower_than_next = diff_y_next < -tolerance

    equal_to_next = np.fabs(diff_y_next) < tolerance

    climbing_up = reduce(lambda l1, l2: l1 + [(l2[0] and l2[2]) or (l1[-1] and not any(l2[:2]))], zip(higher_than_prev, lower_than_prev, equal_to_next), [False])[1:]
    climbing_down = reduce(lambda l1, l2: l1 + [(l2[0] and l2[2]) or (l1[-1] and not any(l2[:2]))], zip(lower_than_prev, higher_than_prev, equal_to_next), [False])[1:]

    # Find maximum crests on left turns only
    max_crest_ids = np.nonzero(np.bitwise_and(np.bitwise_not(direction), higher_than_prev * higher_than_next + higher_than_prev * climbing_up))[0]
    min_crest_ids = np.nonzero(np.bitwise_and(direction, lower_than_prev * lower_than_next + lower_than_prev * climbing_down))[0]
    
    if len(max_crest_ids) > 0:
        max_crest = max_crest_ids[np.argmax(vertices[max_crest_ids, 1])]
    else:
        max_crest = None
    
    if len(min_crest_ids) > 0:
        min_crest = min_crest_ids[np.argmin(vertices[min_crest_ids, 1])]
    else:
        min_crest = None
    
    return max_crest_ids, min_crest_ids, max_crest, min_crest


def _get_crest_cuts(vertices_info, crest_ids):    
    """ Finds the positional indices of the intersection vertices closest to each crest point.

    Parameters 
    ----------
    vertices_info : numpy.ndarray
        An array containing the coordinates of the polygon vertices with additional information of each vertex.
    
    Returns
    -------
    closest_ids : list
        A list of tuples with the index of the crest point, the positional index of the two intersection vertices
        that are closest to that, and their corresponding ray index. 

    """
    n_vertices = vertices_info.shape[0]
    closest_ids = []

    for i in crest_ids:
        shifted_indices = np.mod(i + np.arange(n_vertices), n_vertices)
        prev_id = np.where(vertices_info[shifted_indices, 2].astype(np.int32) >= 0)[0][-1]
        prev_id = shifted_indices[prev_id]

        shifted_indices = np.mod(i + 1 + np.arange(n_vertices), n_vertices)
        next_id = np.where(vertices_info[shifted_indices, 2].astype(np.int32) >= 0)[0][0]
        next_id = shifted_indices[next_id]

        r = int(vertices_info[prev_id, 2])

        closest_ids.append((i, r, prev_id, next_id))

    return closest_ids


def _get_self_intersections(vertices):    
    """ Computes the rays formulae of all edges to identify self-intersections later.
    This will iterate over all edges to generate the ray formulae.

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


def _sort_rays(rays_formulae, max_crest_y, tolerance=1e-4):
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
    
    sorted_rays_formulae = [rays_formulae[i] for i in np.array(list(map(lambda ray: math.fabs(ray[0][1] - max_crest_y), rays_formulae))).argsort()]

    curr_id = len(sorted_rays_formulae) - 1
    while curr_id > 0:
        curr_y = sorted_rays_formulae[curr_id][0][1]
        same_ray = any(filter(lambda r:  math.fabs(r[0][1] - curr_y) < tolerance, sorted_rays_formulae[:curr_id]))
        # If there is at least one ray close (< tolerance) to this, remove the current ray
        if same_ray:
            sorted_rays_formulae.pop(curr_id)
        curr_id -= 1

    return sorted_rays_formulae


def _find_intersections(vertices, rays_formulae, tolerance=1e-4):
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


def _sort_ray_cuts(vertices_info, rays_formulae):
    """ Adds the positional ordering of the intersection vertices that are on a ray.
    It also assigns their corresponding sign according to the direction of the polygon when it is walked from left to right.
    
    Parameters 
    ----------
    vertices_info : numpy.ndarray
        An array containing the coordinates of the polygon vertices with additional information about each vertex.
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray.
    
    Returns
    -------
    vertices_info : numpy.ndarray
        The set of vetices coordinates with the updated information about the position of intersection vertices.

    """
    all_idx_per_ray = []
    all_ord_per_ray = []
    all_sym_per_ray = []
    for k, ((_, rs_y, _), _, _) in enumerate(rays_formulae):
        sel_k = np.nonzero(vertices_info[:, -1].astype(np.int32) == k)[0]
        if len(sel_k) == 0:
            continue
        
        # determine the intersection's symbol using the y coordinate of the previous point of each intersection (sel_k - 1)
        inter_symbols = (vertices_info[sel_k - 1, 1] < rs_y) * 2 - 1
        
        rank_ord = np.empty(len(sel_k))
        rank_ord[np.argsort(vertices_info[sel_k, 0])] = list(range(len(sel_k)))

        all_idx_per_ray += list(rank_ord)
        all_ord_per_ray += list(sel_k)
        all_sym_per_ray += list(inter_symbols)

    vertices_info = np.hstack((vertices_info, np.zeros((vertices_info.shape[0], 2))))
    # Fourth column contains the symbol of the cut, and the sixth column the index of that cut on the corresponding ray
    vertices_info[all_ord_per_ray, -2] = all_sym_per_ray
    vertices_info[all_ord_per_ray, -1] = all_idx_per_ray

    return vertices_info


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
    vertices_info : numpy.ndarray
        The set of vetices coordinates with the updated information about the position of intersection vertices.

    """
    vertices_info = []
    last_j = 0

    for i in np.unique(valid_edges[:, 0]):
        vertices_info.append(np.hstack((vertices[last_j:i+1, :], -np.ones((i-last_j+1, 1)))))
        sel_i = np.nonzero(valid_edges[:, 0] == i)[0]
        sel_r = valid_edges[sel_i, 2]

        ord_idx = np.argsort(t_coefs[sel_i])
        
        sel_i = sel_i[ord_idx]
        sel_r = sel_r[ord_idx]

        vertices_info.append(np.hstack((cut_coords[sel_i], sel_r.reshape(-1, 1))))
        last_j = i + 1

    n_vertices = vertices.shape[0]
    vertices_info.append(np.hstack((vertices[last_j:, :], -np.ones((n_vertices-last_j, 1)))))
    vertices_info = np.vstack(vertices_info)

    return vertices_info


def subdivide_selfoverlapping(vertices):
    """ Divide a self-overlapping polygon into non self-overlapping polygons.
    This implements the algorithm proposed in [1].

    Parameters 
    ----------
    vertices : numpy.ndarray
        A two-dimensional array containing the x and y coordinates of the polygon vertices.
    
    Returns
    -------
    sub_polys : list
        A list of numpy.ndarrays with the coordinates of the non self-overlapping polygons obtained from sub-dividing the original polygon.

    References
    ----------
    .. [1] Uddipan Mukherjee. (2014). Self-overlapping curves:
           Analysis and applications. Computer-Aided Design, 46, 227-232.
           :DOI: https://doi.org/10.1016/j.cad.2013.08.037
    """
    max_crest_ids, min_crest_ids, max_crest, min_crest = _get_crest_ids(vertices, tolerance=2*np.finfo(np.float32).eps)

    if max_crest is None and min_crest is None:
        # If the polygon does not have any crest point, it is because it is not self-overlapping.
        return [vertices]

    rays_max = _compute_rays(vertices, max_crest_ids, epsilon=-1e-4)
    rays_min = _compute_rays(vertices, min_crest_ids, epsilon=1e-4)
    rays_formulae = _sort_rays(rays_max + rays_min, vertices[:, 1].max(), tolerance=2e-4)
    self_inter_formulae = _get_self_intersections(vertices)

    cut_coords, valid_edges, t_coefs = _find_intersections(vertices, rays_formulae + self_inter_formulae, tolerance=2*np.finfo(np.float32).eps)
    vertices_info = _merge_new_vertices(vertices, cut_coords, valid_edges, t_coefs)
    vertices_info = _sort_ray_cuts(vertices_info, rays_formulae)

    # Get the first point at the left of the crest point
    new_max_crest_ids, new_min_crest_ids, _, _ = _get_crest_ids(vertices_info, tolerance=2*np.finfo(np.float32).eps)
    new_max_crest_cuts = _get_crest_cuts(vertices_info, new_max_crest_ids)
    new_min_crest_cuts = _get_crest_cuts(vertices_info, new_min_crest_ids)

    left_idx, right_idx = _get_root_indices(vertices_info)

    # The root is left valid by construction.
    # Therefore, the right validity of the root cut is checked and then all the possible valid cuts are computed.
    visited = {}
    _, root_id = _check_validity(left_idx, right_idx, vertices_info, new_max_crest_cuts, new_min_crest_cuts, visited, check_left=False)

    # Update the visited dictionary to leave only valid paths
    polygon_is_valid = _traverse_tree(root_id, visited)

    if not polygon_is_valid:
        # If the polygon cannot be sub-divided into polygons that are not self-overlapping, return the original polygon
        return [vertices]

    # Perform a single immersion on the validity tree to get the first valid path that cuts the polygon into non self-overlapping sub-polygons
    for k in list(visited.keys()):
        if visited[k][0]:
            visited[k][0] = None
        else:
            del visited[k]

    sub_polys_ids = []
    _, sub_poly = _immerse_valid_tree(root_id, visited, vertices_info, sub_polys_ids)

    fig, ax = plt.subplots()
    ax.plot(vertices_info[:, 0], vertices_info[:, 1], 'r:')
    ax.plot([vertices_info[-1, 0], vertices_info[0, 0]], [vertices_info[-1, 1], vertices_info[0, 1]], 'r:')
    ax.plot(vertices_info[:, 0], vertices_info[:, 1], 'r.')

    for r, ((_, ry, _), _, _) in enumerate(rays_formulae):
        ax.plot([np.min(vertices_info[:, 0]), np.max(vertices_info[:, 0])], [ry, ry], 'b-.')
        for a_r in np.where(vertices_info[:, 2].astype(np.int32) == r)[0]:
            ax.text(vertices_info[a_r, 0] - 1e-3, vertices_info[a_r, 1] + 1e-3, '%s%i%s' % (chr(97 + r), int(vertices_info[a_r, 4])+1, '\'' if vertices_info[a_r, 3] < 0.5 else ''), fontsize=12)

    plt.show()

    # Add the root cut of the immersion tree
    n_vertices = vertices_info.shape[0]
    shifted_indices = np.mod(left_idx + np.arange(n_vertices), n_vertices)
    r = right_idx - left_idx + 1 + (0 if right_idx > left_idx else n_vertices)
    sub_poly = [shifted_indices[:r]] + sub_poly
    sub_polys_ids.insert(0, np.concatenate(sub_poly, axis=0))

    polys = []
    for poly_ids in sub_polys_ids:
        polys.append(vertices_info[poly_ids, :2])

    return polys