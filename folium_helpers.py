import os
import datetime as dt
import csv
from chardet import detect
import random
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# geo
import shapely
from shapely.geometry import Point, MultiPolygon, Polygon, MultiPoint
import geopandas as gpd
import folium
import json

from grid_system import ROADS, LANDUSE, pm_grid_from_file, TRAFFIC
from constants_jo import *
from ndata import STATIC_POINTS
from neval import OUTPUT_DIRS

from kriging import PLOT_DIRS as KRIGING_PLOT_DIRS
from rfsi import PLOT_DIRS as RFSI_PLOT_DIRS
from ntrain import PLOT_DIRS as CNN_PLOT_DIRS

CENTRE_COORDS = {
    'leon':(21.114657, -101.672406),
    'guadalajara': (20.675007, -103.354761),
    'edinburgh_2018': (55.942627, -3.188735),
    'edinburgh': (55.942627, -3.188735),
    'delhi': (0.0, 0.0)
    }

GIS_DIRECTORIES = {
    'leon': leon_gis_dir,
    'guadalajara': guadalajara_gis_dir,
    'edinburgh_2018': edinburgh_gis_dir,
    'edinburgh': edinburgh_2018_gis_dir,
    'delhi': delhi_gis_dir
    }

# ROAD_COLORS = {0:'#edf8fb', # ped/cycle
#                1:'#bfd3e6', # resi/service
#                2:'#9ebcda', # unclassified
#                3:'#8c96c6', # tertiary
#                4:'#8856a7', # secondary
#                5:'#810f7c', # primary/trunk
#                None: 'black'}

ROAD_COLORS = {0:'#ffffb2',
               1:'#fed976',
               2:'#feb24c',
               3:'#fd8d3c',
               4:'#f03b20',
               5:'#bd0026',
               None: 'black'}

LANDUSE_COLORS = {0:'#7fc97f', # green
                  1:'#beaed4', # resi
                  2:'#fdc086'} # commercial

POINT_COLORS = ['#FFFFB2', '#FED976', '#FEB24C','#FD8D3C', '#FC4E2A', '#E31A1C', '#B10026']

def create_map(project_name, tiles='Stamen Toner', zoom_start=13):
    m = folium.Map(
        location=CENTRE_COORDS[project_name],
        tiles='Stamen Toner',
        zoom_start=zoom_start
        )
    
    return m

def id_to_numpy(cell_id, grid_dim):
    return np.unravel_index(cell_id-1, grid_dim, order='F')

def unpack_id(coord, dim=0):
    return coord[dim]

def add_grid_data_to_gdf(gdf, grid_data, new_col, zero_nan=True, negative_nan=True):
    '''
    adds new column containing data from grid to geodataframe
    based on "numpy" coordinate column
    if zero_nan true, sets 0.0 values to nan (useful for chloropleth)
    '''
    gdf[new_col] = np.nan
    for i in range(grid_data.shape[0]):
        for j in range(grid_data.shape[1]):
            coord = (i, j)
            index = gdf[gdf['numpy'] == coord].index.values[0]
            # if grid_data[i,j] > 0 or not(negative_nan):
            #     gdf.loc[index, new_col] = grid_data[i,j]
            # if grid_data[i,j] != 0 or not(zero_nan):
            #     gdf.loc[index, new_col] = grid_data[i,j]
            # if grid_data[i,j] > 0 or not(zero_nan):
            #     gdf.loc[index, new_col] = grid_data[i,j]
            if grid_data[i, j] > 0:
                gdf.loc[index, new_col] = grid_data[i,j]
    return gdf

def create_pm_map_object(grid_filename, geojson_filepath, timestep=None, static_points=None, bins=7, mask_negative_values=True,
                         fill_opacity=0.7, line_opacity=0.3, convert_coords=True, fill_color='YlOrRd',  nan_fill_color='grey', nan_fill_opacity=0.0):
    '''
    creates chloropleth of PM values to add to a folium map
    '''
    if timestep is None: # obtain pm grid data
        pm_grid = np.load(grid_filename)
    else:
        pm_grid = np.load(grid_filename)[timestep] 
    grid_dim = pm_grid.shape
    
    grid_df = gpd.read_file(geojson_filepath)
    # add grid indices to data
    grid_df['numpy'] = grid_df.apply(lambda x: id_to_numpy(x['id'], grid_dim), axis=1)
    grid_df['numpy_0'] = grid_df.apply(lambda x: unpack_id(x['numpy'], dim=0), axis=1)
    grid_df['numpy_1'] = grid_df.apply(lambda x: unpack_id(x['numpy'], dim=1), axis=1)
    if convert_coords: # set to true if grid not displaying in correct location
        for col in ('left','top','right','bottom'):
            grid_df[col] = grid_df[col]/100000
    
    # add pm data to df
    grid_df = add_grid_data_to_gdf(grid_df, pm_grid, 'pm', zero_nan=True)
    print(grid_df['pm'].max())
    
    chlr = folium.Choropleth(
        geo_data=geojson_filepath,
        data=grid_df,
        columns=['id', 'pm'],
        key_on='properties.id',
        legend_name = 'PM level',
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        fill_color=fill_color,
        nan_fill_color = nan_fill_color,
        nan_fill_opacity = nan_fill_opacity,
        bins=bins
    )
    
    if static_points is not None:
        points = []
        static_coords = []
        x_coords, y_coords = static_points[0], static_points[1]
        for k in range(len(x_coords)):
            coord = (x_coords[k], y_coords[k])
            index = grid_df[grid_df['numpy'] == coord].index.values[0]
            static_coords.append(grid_df.loc[index, 'geometry'])
        for coord in static_coords:
            points.append(folium.Marker([coord.centroid.y, coord.centroid.x]))
        return chlr, points
    
    return chlr

def create_pm_map_object_wrapper(project_name, cell_size, interval, timestep,
                                 coord_idx = 0, pm = 'pm2_5',convert_coords=True):
    '''
    wrapper for create pm map object for collected personal data
    '''
    label='personal'
    data_dir = data_dir = project_mapping[project_name][2]
    gis_dir = project_mapping[project_name][4]
    grid_filename = os.path.join(data_dir, 'grid', f'{project_name}_{label}_{cell_size}m_{interval}min_{pm}.npy')
    geojson_filepath = os.path.join(gis_dir, f'{project_name}_grid_{cell_size}.geojson')
    
    static_grid = pm_grid_from_file(os.path.join(data_dir, 'grid', f'{project_name}_static_{cell_size}m_{interval}min_pm2_5.npy'))
    static_points = static_grid[coord_idx].nonzero()
    
    return create_pm_map_object(grid_filename, geojson_filepath, timestep, convert_coords=convert_coords, static_points=static_points)

def create_histogram_map_object(grid_filename, geojson_filepath, static_points=None, bins=7,
                                fill_opacity=0.7, line_opacity=0.5, convert_coords=True, fill_color='BuPu',  nan_fill_color='grey', nan_fill_opacity=0.0):
    '''
    creates chloropleth showing the number of grid cells with data 
    over all time intervals
    '''
    grid = np.load(grid_filename) # obtain pm grid data
    T = grid.shape[0]
    grid_dim = grid.shape[1], grid.shape[2]
    
    # create hist grid
    hist_grid = np.zeros(grid_dim)
    for t in range(T):
        hist_grid += np.where(grid[t] > 0, 1, 0)
    
    grid_df = gpd.read_file(geojson_filepath)
    # add grid indices to data
    grid_df['numpy'] = grid_df.apply(lambda x: id_to_numpy(x['id'], grid_dim), axis=1)
    grid_df['numpy_0'] = grid_df.apply(lambda x: unpack_id(x['numpy'], dim=0), axis=1)
    grid_df['numpy_1'] = grid_df.apply(lambda x: unpack_id(x['numpy'], dim=1), axis=1)
    if convert_coords: # set to true if grid not displaying in correct location
        for col in ('left','top','right','bottom'):
            grid_df[col] = grid_df[col]/100000
    
    # add pm data to df
    grid_df = add_grid_data_to_gdf(grid_df, hist_grid, 'count', zero_nan=True)
    
    chlr = folium.Choropleth(
        geo_data=geojson_filepath,
        data=grid_df,
        columns=['id', 'count'],
        key_on='properties.id',
        legend_name = 'Number of timesteps',
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        fill_color=fill_color,
        nan_fill_color = nan_fill_color,
        nan_fill_opacity = nan_fill_opacity,
        bins=bins
    )
    
    if static_points is not None:
        points = []
        static_coords = []
        x_coords, y_coords = static_points[0], static_points[1]
        for k in range(len(x_coords)):
            coord = (x_coords[k], y_coords[k])
            index = grid_df[grid_df['numpy'] == coord].index.values[0]
            static_coords.append(grid_df.loc[index, 'geometry'])
        for coord in static_coords:
            points.append(folium.Marker([coord.centroid.y, coord.centroid.x]))
        return chlr, points
    
    return chlr

def create_histogram_map_object_wrapper(project_name, cell_size, interval, convert_coords=True, coord_idx=None, static_points=None, fill_color='BuPu'):
    '''
    wrapper for create histogram map object
    '''
    label='personal'
    model_name = f'{project_name}_cnn_{interval}min_{cell_size}'
    data_dir = project_mapping[project_name][2]
    gis_dir = project_mapping[project_name][4]
    grid_filename = os.path.join(data_dir, 'grid', f'{project_name}_{label}_{cell_size}m_{interval}min_pm2_5.npy')
    geojson_filepath = os.path.join(gis_dir, f'{project_name}_grid_{cell_size}.geojson')
    
    if static_points is None and coord_idx is None:
        static_points = np.load(STATIC_POINTS[model_name])
    elif static_points is None:
        static_grid = pm_grid_from_file(os.path.join(data_dir, 'grid', f'{project_name}_static_{cell_size}m_{interval}min_pm2_5.npy'))
        static_points = static_grid[coord_idx].nonzero()
    
    return create_histogram_map_object(grid_filename, geojson_filepath, convert_coords=convert_coords, static_points=static_points, fill_color=fill_color)

def create_road_map_object(geojson_filepath):
    '''
    creates coloured roads to add to a folium map
    '''
    roads_df = gpd.read_file(geojson_filepath)
    roads_df['roadid']= roads_df['fclass'].map(ROADS) # augment gdf with road data
    #roads_df['fid'] = roads_df['fid'].astype(np.int32)
    roads_df = roads_df.dropna()
    roads_df['roadid']= roads_df['roadid'].astype(np.int32)
    
    def style_function(feature): # style function for folium
        if feature['properties']['roadid'] is None:
            opacity = 0.0 # make non classified rows transparent
        else:
            opacity = 1.0
        color = ROAD_COLORS[feature['properties']['roadid']]
        return {'lineColor':color,
                'polyLineColor':color,
                'fillColor':color,
                'color':color,
                'fillOpacity': opacity}
    
    rd = folium.GeoJson(roads_df, style_function=style_function)
    return rd

def create_landuse_map_object(geojson_filepath):
    '''
    creates coloured landuse polygons to add to a folium map
    '''
    landuse_df = gpd.read_file(geojson_filepath)
    landuse_df['landuseid']= landuse_df['fclass'].map(LANDUSE) # augment gdf with road data
    #roads_df['fid'] = roads_df['fid'].astype(np.int32)
    landuse_df = landuse_df.dropna()
    landuse_df['landuseid']= landuse_df['landuseid'].astype(np.int32)
    
    def style_function(feature): # style function for folium
        if feature['properties']['landuseid'] is None:
            opacity = 0.0 # make non classified rows transparent
        else:
            opacity = 0.6
        color = LANDUSE_COLORS[feature['properties']['landuseid']]
        return {'lineColor':color,
                'polyLineColor':color,
                'fillColor':color,
                'color':color,
                'fillOpacity': opacity}
    
    landuse = folium.GeoJson(landuse_df, style_function=style_function)
    return landuse

def create_junction_map_object(geojson_filepath):
    '''
    creates junction markers to add to folium map
    '''
    traffic_df = gpd.read_file(geojson_filepath)
    traffic_df['trafficid']= traffic_df['fclass'].map(TRAFFIC)
    traffic_df['trafficid'] = traffic_df['trafficid'].fillna(0)
    points = []
    coords = []
    for k in range(len(traffic_df)):
        if traffic_df.loc[k, 'trafficid'] == 1:
            coords.append(traffic_df.loc[k, 'geometry'])
    for coord in coords:
        points.append(folium.Marker([coord.centroid.y, coord.centroid.x], icon=folium.Icon(color='red', icon='glyphicon glyphicon-asterisk')))
    return points

# def create_points_map_object(pm_grid, geojson_filepath, bins=None, convert_coords=True):
#     '''
#     creates coloured points to add to a folium map with grid
#     '''
#     grid_df = gpd.read_file(geojson_filepath)
#     grid_df['numpy'] = grid_df.apply(lambda x: id_to_numpy(x['id'], grid_dim), axis=1)
#     grid_df['numpy_0'] = grid_df.apply(lambda x: unpack_id(x['numpy'], dim=0), axis=1)
#     grid_df['numpy_1'] = grid_df.apply(lambda x: unpack_id(x['numpy'], dim=1), axis=1)
#     if convert_coords: # set to true if grid not displaying in correct location
#         for col in ('left','top','right','bottom'):
#             grid_df[col] = grid_df[col]/100000
            
#     if bins is None: # default into 7 evenly spaced bin values
#         bins = list(np.linspace(pm_grid.min(), pm_grid.max(), num=7))
#     color_dict = {(bins[i], bins[i+1]) for }
    
#     grid_df['bin']= roads_df['fclass'].map(ROADS) # augment gdf with bin class
#     roads_df['fid'] = roads_df['fid'].astype(np.int32)
#     roads_df['roadid']= roads_df['roadid'].astype(np.int32)
    
#     def style_function(feature): 
#         color = colors[feature['properties']['roadid']]
#         return {'lineColor':color,
#                 'polyLineColor':color,
#                 'fillColor':color,
#                 'color':color,
#                 'fillOpacity': opacity}
    
#     rd = folium.GeoJson(roads_df, style_function=style_function)
#     return rd

def kriging_mapping(location, interval, cell_size, i=0, plot_grid='output', vm='gaussian', kriging_type='ordinary', dataset='test', bins=None):
    gis_dir = project_mapping[location][4]
    model_name = f'{location}_cnn_{interval}min_{cell_size}'
    static_points = np.load(STATIC_POINTS[model_name])
    
    if bins is None:
        bins = 7
    
    geojson_filepath = os.path.join(gis_dir, f'{location}_grid_{cell_size}.geojson')
    target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{dataset}_{i}_y.npy')
    output_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{dataset}_{i}_out.npy')
    
    m = create_map(location)
    if plot_grid == 'output':
        print(f'Plotting {output_grid_filename}')
        chlor, points = create_pm_map_object(output_grid_filename, geojson_filepath, timestep=0, static_points=static_points, bins=bins)
    else:
        print(f'Plotting {target_grid_filename}')
        chlor, points = create_pm_map_object(target_grid_filename, geojson_filepath, timestep=0, static_points=static_points, bins=bins)
    
    chlor.add_to(m)
    for p in points:
        p.add_to(m)
        
    return m

def rfsi_mapping(location, interval, cell_size, i=0, plot_grid='output', dataset='test', rfsi_t=False, avg_static_pm=False, bins=None):
    gis_dir = project_mapping[location][4]
    if rfsi_t:
        model_name = f'{location}_lstm_{interval}min_{cell_size}'
    else:
        model_name = f'{location}_cnn_{interval}min_{cell_size}'
    static_points = np.load(STATIC_POINTS[model_name])
    
    if avg_static_pm:
        suffix = '_avg_pm'
    else:
        suffix = ''
        
    if bins is None:
        bins = 7
    
    geojson_filepath = os.path.join(gis_dir, f'{location}_grid_{cell_size}.geojson')
    if rfsi_t:
        target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_rfsi_t{suffix}_{dataset}_{i}_y.npy')
        output_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_rfsi_t{suffix}_{dataset}_{i}_out.npy')
    else:
        target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{dataset}_{i}_y.npy')
        output_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{dataset}_{i}_out.npy')

    m = create_map(location)
    if plot_grid == 'output':
        print(f'Plotting {output_grid_filename}')
        chlor, points = create_pm_map_object(output_grid_filename, geojson_filepath, static_points=static_points, bins=bins)
    else:
        print(f'Plotting {target_grid_filename}')
        chlor, points = create_pm_map_object(target_grid_filename, geojson_filepath, static_points=static_points, bins=bins)
    
    chlor.add_to(m)
    for p in points:
        p.add_to(m)
        
    return m

def cnn_mapping(location, interval, cell_size, mr, epoch, i=0, plot_grid='output', dataset='test', bins=None):
    gis_dir = project_mapping[location][4]
    model_name = f'{location}_cnn_{interval}min_{cell_size}'
    static_points = np.load(STATIC_POINTS[model_name])
    
    if bins is None:
        bins = 7
    
    geojson_filepath = os.path.join(gis_dir, f'{location}_grid_{cell_size}.geojson')
    target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_epoch{epoch}_{mr}_{dataset}_{i}_y.npy')
    pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_epoch{epoch}_{mr}_{dataset}_{i}_out.npy')

    m = create_map(location)
    if plot_grid == 'output':
        print(f'Plotting {pred_grid_filename}')
        chlor, points = create_pm_map_object(pred_grid_filename, geojson_filepath, static_points=static_points, timestep=0, bins=bins)
    else:
        print(f'Plotting {target_grid_filename}')
        chlor, points = create_pm_map_object(target_grid_filename, geojson_filepath, static_points=static_points, timestep=0, bins=bins)
    
    chlor.add_to(m)
    for p in points:
        p.add_to(m)
        
    return m

def scatter_plot_actual_predicted(y_grid, out_grid, plot_filename=None, figsize=(4,4), multiple_grids=False, plot_min=None, plot_max=None):
    '''
    plots a scatter plot of predicted PM values (x axis) vs actual (y axis)
    y_grid, out_grid must be of shape h x w
    '''
    nonzero_points = y_grid.nonzero()
    x = []
    y = []
    if multiple_grids:
        for p in range(len(nonzero_points[0])):
            i, j, k = nonzero_points[0][p], nonzero_points[1][p], nonzero_points[2][p]
            y.append(y_grid[i,j,k])
            x.append(out_grid[i,j,k])
    else:
        for p in range(len(nonzero_points[0])):
            i, j = nonzero_points[0][p], nonzero_points[1][p]
            y.append(y_grid[i,j])
            x.append(out_grid[i,j])
    
    if plot_min is None:
        new_min = min(min(y), min(x))-0.1
    else:
        new_min = plot_min
        
    if plot_max is None:
        new_max = max(max(y), max(x))+0.1
    else:
        new_max = plot_min
    
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([new_min,new_max], [new_min,new_max], color='slategray')
    plot = sns.scatterplot(x=x, y=y, ax=ax, color='tomato')
    ax.set_xlim(new_min, new_max)
    ax.set_ylim(new_min, new_max)
    ax.set_xlabel('Predicted PM2.5 values')
    ax.set_ylabel('Actual PM2.5 values')
    
    if plot_filename is not None:
        fig = plot.get_figure()
        fig.savefig(plot_filename)
        
def kriging_actual_predicted(location, interval, cell_size, i=None, vm='gaussian', kriging_type='ordinary', dataset='test', plot_max=None, plot_min=None):
    model_name = f'{location}_cnn_{interval}min_{cell_size}'
    if i is None:
        multiple_grids = True
        files = os.listdir(OUTPUT_DIRS[model_name])
        print(len(files))
        files = [f for f in files if model_name in f if kriging_type in f if vm in f if dataset in f if 'kriging' in f]
        target_files = sorted([f for f in files if '_y.npy' in f])
        pred_files = sorted([f for f in files if '_out.npy' in f])
        y_grids = []
        out_grids = []
        for f in range(len(target_files)):
            target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], target_files[f])
            pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], pred_files[f])
            plot_filename = os.path.join(KRIGING_PLOT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{dataset}_all_scatter.png')
            y_grids.append(np.load(target_grid_filename)[0])
            out_grids.append(np.load(pred_grid_filename)[0])
        y_grid = np.stack(y_grids)
        out_grid = np.stack(out_grids)
    else:
        multiple_grids = False
        target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{dataset}_{i}_y.npy')
        pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{dataset}_{i}_out.npy')
        plot_filename = os.path.join(KRIGING_PLOT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{dataset}_{i}_scatter.png')
        y_grid = np.load(target_grid_filename)[0]
        out_grid = np.load(pred_grid_filename)[0]
        
    scatter_plot_actual_predicted(y_grid, out_grid, plot_filename=plot_filename, multiple_grids=multiple_grids)
    
def rfsi_actual_predicted(location, interval, cell_size, i=None, dataset='test', avg_static_pm=False, plot=True, plot_max=None, plot_min=None):
    model_name = f'{location}_cnn_{interval}min_{cell_size}'
    if avg_static_pm:
        suffix = '_avg_pm'
    else:
        suffix = ''
        
    if i is None:
        multiple_grids = True
        files = os.listdir(OUTPUT_DIRS[model_name])
        files = [f for f in files if model_name in f if dataset in f if f'rfsi{suffix}' in f]
        target_files = sorted([f for f in files if '_y.npy' in f])
        pred_files = sorted([f for f in files if '_out.npy' in f])
        y_grids = []
        out_grids = []
        for f in range(len(target_files)):
            target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], target_files[f])
            pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], pred_files[f])
            plot_filename = os.path.join(RFSI_PLOT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{dataset}_all_scatter.png')
            y_grids.append(np.load(target_grid_filename))
            out_grids.append(np.load(pred_grid_filename))
        y_grid = np.stack(y_grids)
        out_grid = np.stack(out_grids)
    else:
        multiple_grids = False
        target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{dataset}_{i}_y.npy')
        pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{dataset}_{i}_out.npy')
        plot_filename = os.path.join(RFSI_PLOT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{dataset}_{i}_scatter.png')
        y_grid = np.load(target_grid_filename)
        out_grid = np.load(pred_grid_filename)
    
    if plot:
        scatter_plot_actual_predicted(y_grid, out_grid, plot_filename=plot_filename, multiple_grids=multiple_grids, plot_max=plot_max, plot_min=plot_min)
    else:
        return y_grid, out_grid
    
def cnn_actual_predicted(location, interval, cell_size, mr, epoch, i=None, dataset='test', plot=True, plot_max=None, plot_min=None):
    model_name = f'{location}_cnn_{interval}min_{cell_size}'
    if i is None:
        multiple_grids = True
        files = os.listdir(OUTPUT_DIRS[model_name])
        files = [f for f in files if model_name in f if dataset in f if f'epoch{epoch}' in f if mr in f]
        target_files = sorted([f for f in files if '_y.npy' in f])
        pred_files = sorted([f for f in files if '_out.npy' in f])
        y_grids = []
        out_grids = []
        for f in range(len(target_files)):
            target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], target_files[f])
            pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], pred_files[f])
            plot_filename = os.path.join(KRIGING_PLOT_DIRS[model_name], f'{model_name}_{dataset}_all_scatter.png')
            y_grids.append(np.load(target_grid_filename)[0])
            out_grids.append(np.load(pred_grid_filename)[0])
        y_grid = np.stack(y_grids)
        out_grid = np.stack(out_grids)
    else:
        multiple_grids = False
        target_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_epoch{epoch}_{mr}_{dataset}_{i}_y.npy')
        pred_grid_filename = os.path.join(OUTPUT_DIRS[model_name], f'{model_name}_epoch{epoch}_{mr}_{dataset}_{i}_out.npy')
        plot_filename = os.path.join(CNN_PLOT_DIRS[model_name], f'{model_name}_{dataset}_{i}_scatter.png')
        y_grid = np.load(target_grid_filename)[0]
        out_grid = np.load(pred_grid_filename)[0]
    
    if plot:
        scatter_plot_actual_predicted(y_grid, out_grid, plot_filename=plot_filename, multiple_grids=multiple_grids, plot_max=plot_max, plot_min=plot_min)
    else:
        return y_grid, out_grid
