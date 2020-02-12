"""
Get Nearest Grid around stations
"""

import pandas as pd
import os
import merge_grid_weather as mg

debug = False

def get_near_grids(row, grid):

    latitude = row['latitude']
    longitude = row['longitude']
    mask1 = (grid['latitude'] >= (latitude - 0.1)) & (grid['latitude'] <= (latitude + 0.1))
    mask2 = (grid['longitude'] >= (longitude - 0.1)) & (grid['longitude'] <= (longitude + 0.1))

    df2 = grid.loc[mask1 & mask2]
    # print(len(df2), df2.index.tolist())
    return df2.index.tolist()


def fill_near_grids(grid, aq):
    aq['grids'] = aq.apply(lambda row: get_near_grids(row, grid),axis=1)

    if debug:
        for idx, row in aq.iterrows():
            print("STATION = {}, {} -> {}".format(idx, len(row['grids']), row['grids']))

    return aq


def get_station_data(city):
    if city == 'bj':
        grid, aq = mg.get_beijing_data()
    elif city == 'ld':
        grid, aq = mg.get_london_data()
    else:
        AssertionError("Cannot load data")

    aq = fill_near_grids(grid, aq)
    return grid, aq


if __name__ == '__main__':
    bj_grid, bj_aq = get_station_data('bj')
    ld_grid, ld_aq = get_station_data('ld')

    print(bj_aq.head(5))
    print(ld_aq.head(5))
