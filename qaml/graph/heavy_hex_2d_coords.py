def d7_2d_coords(nodes_1_indexed=True, edges_tuples=True):
    coords = {}

    # Full rows (y = 1,3,5,7,9,11,13)
    # Odd x carry the small numbers decreasing left→right.
    # Even x carry the large numbers decreasing left→right.
    odd_rows = [1, 3, 5, 7, 9, 11, 13]
    for i, y in enumerate(odd_rows):
        small_base = 6 + 7 * i  # 6,13,20,27,34,41,48 at x = 1,3,5,7,9,11,13
        big_base = 78 + 6 * i  # 78,84,90,96,102,108,114 at x = 2,4,6,8,10,12
        for j, x in enumerate([1, 3, 5, 7, 9, 11, 13]):
            coords[small_base - j] = [x, y]
        for j, x in enumerate([2, 4, 6, 8, 10, 12]):
            coords[big_base - j] = [x, y]

    # Exta leftover nodes
    coords.update({52: [2, 2], 51: [6, 2], 50: [10, 2], 49: [13, 2]})
    coords.update({56: [1, 4], 55: [4, 4], 54: [8, 4], 53: [12, 4]})
    coords.update({60: [2, 6], 59: [6, 6], 58: [10, 6], 57: [13, 6]})
    coords.update({64: [1, 8], 63: [4, 8], 62: [8, 8], 61: [12, 8]})
    coords.update({68: [2, 10], 67: [6, 10], 66: [10, 10], 65: [13, 10]})
    coords.update({72: [1, 12], 71: [4, 12], 70: [8, 12], 69: [12, 12]})

    if nodes_1_indexed:
        # Increment all node labels to be index-1 for Julia
        coords = {k+1: v for k,v in coords.items()}
    if edges_tuples:
        coords = {k: tuple(v) for k, v in coords.items() }
    return coords

d5_2d_coords_list = [(0, (3, 1)), (1, (2, 2)), (2, (1, 3)), (3, (1, 5)), (4, (1, 7)), (5, (5, 1)), (6, (5, 3)), (7, (4, 4)), (8, (3, 5)), (9, (3, 7)), (10, (7, 1)), (11, (7, 3)), (12, (6, 4)), (13, (6, 6)), (14, (5, 7)), (15, (9, 1)), (16, (9, 3)), (17, (9, 5)), (18, (8, 6)), (19, (8, 8)), (20, (11, 1)), (21, (11, 3)), (22, (11, 5)), (23, (11, 7)), (24, (10, 8)), (25, (4, 1)), (26, (3, 3)), (27, (2, 6)), (28, (6, 2)), (29, (5, 5)), (30, (4, 7)), (31, (8, 1)), (32, (8, 4)), (33, (7, 7)), (34, (10, 2)), (35, (10, 6)), (36, (9, 8)), (37, (2, 1)), (38, (2, 3)), (39, (1, 4)), (40, (1, 6)), (41, (5, 2)), (42, (4, 3)), (43, (4, 5)), (44, (3, 6)), (45, (7, 2)), (46, (7, 4)), (47, (6, 5)), (48, (6, 7)), (49, (9, 2)), (50, (9, 4)), (51, (9, 6)), (52, (8, 7)), (53, (11, 2)), (54, (11, 4)), (55, (11, 6)), (56, (11, 8))]
d5_2d_coords = {mapping[0]+1: (mapping[1][0]+1, mapping[1][1]+1) for mapping in d5_2d_coords_list}