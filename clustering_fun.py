#!/usr/bin/env python3

import sys
import csv
import random
import math
import location
from kmeans import KMeans
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Clustering Fun\n")

    verbose = False
    k = None
    plot = False
    num_points = None
    csv_path = None
    lim_distance = None
    argCount = len(sys.argv)
    args = sys.argv
    locations = None

    for i, arg in enumerate(args):
        if i == 0:
            continue # ignore script name

        if arg.startswith("-"):
            if arg.startswith('-k='):
                k = int(arg.split('=', 1)[1])
            elif arg.startswith('-n='):
                num_points = int(arg.split('=', 1)[1])
            elif arg.startswith('-l='):
                lim_distance = int(arg.split('=', 1)[1])
            elif arg == "-v":
                verbose = True
            elif arg == "-p":
                plot = True
            else:
                print(f"Unknown argument: {arg}")
                print_usage()
                exit(1)
        elif not csv_path:
            csv_path = arg
        else:
            print(f"Too many parameters!")
            print_usage()
            exit(1)

    if not csv_path and not num_points or not k:
        print(f"Too few parameters!")
        print_usage()
        exit(1)

    if verbose:
        print_options(verbose, csv_path, k, plot, lim_distance)

    if csv_path:
        locations = read_csv(csv_path)
        if verbose:
            print(f"\nRead {len(locations)} locations from {csv_path}")
    else:
        locations = generate_points_gauss(num_points, k)

    
    
    print(f"\nSetting up K-Means Classifier:")
    km = KMeans(locations, k, verbose, dist_limit=lim_distance)

    print(f"Finding best initial center points with K-Means++")
    centers = np.array(np.empty)
    min_var = None
    for i in range(8):
        tmp_centers, variance = km.init_centers()
        if verbose:
            print(f"  Iteration {i}, variance = {variance} km")
        if min_var is None or variance < min_var:
            min_var = variance
            centers = np.copy(tmp_centers)
            

    if plot and verbose:
        plot_locations(locations, k, centers, img_tag=f"{'' if not csv_path else csv_path+'_'}initial_state")

    centers, locations = km.classify(centers)

    if plot:
        plot_locations(locations, k, centers, img_tag=f"{'' if not csv_path else csv_path+'_'}")

    print(f"\nClass centers:")
    for idx, c in enumerate(centers):
        print(f'  Class = {idx}:')
        print(f'    Center = {c}')
        print(f'    Farthest point from center = {max([loc.class_distance for loc in locations if loc.classification == idx])} km')
        print(f'    Attribute1 = {any(loc.attr1 for loc in locations if loc.classification == idx)} (Logical OR)')
        print(f'    Attribute2 = {any(loc.attr2 for loc in locations if loc.classification == idx)} (Logical OR)')
        print()

    if lim_distance:
        print(f"\nWith distance limit of {lim_distance} km:")
        print(f"  {len([loc for loc in locations if loc.classification == None])} locations are left unclassified")


def read_csv(csv_path):
    locs=[]

    with open(csv_path, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            locs.append(location.Location(row))
        
    return locs


def generate_points_gauss(num_points, k):
    print(f"Randomly generating {k} gaussian clusters with {num_points} total points:")

    group_num = float(num_points) / k
    locations = []
    for i in range(k):
        c = (random.uniform(-90, 90), random.uniform(-180, 180))
        s = random.uniform(3.0, 18.0)
        x = []
        print(f"Randomly Generated Center {i}: [{c[0]}, {c[1]}]")

        while len(x) < group_num:
            a,b = np.array([np.random.normal(c[0],s), np.random.normal(c[1],s)])
            if b < -180:
                b += 360
            elif b > 180:
                b -= 360

            row = {
                "Name" : f"TestPoint [{i}]",
                "Latitude" : a,
                "Longitude" : b,
                "Attribute1" : random.random() > 0.999,
                "Attribute2" : random.random() > 0.999 
            }
            x.append(location.Location(row))
        locations.extend(x)

    return locations


def plot_locations(locations, k, centers=None, img_tag=None):
    
    img_file_name = 'kpp_%s_K=%s.png' % (
        img_tag if img_tag else '',
        str(k)
    )
    print(f"Plotting {img_file_name} ...")
    fig = plt.figure(figsize=(50,25))
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, k)]
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    for i in range(k):
        data_points_x = np.array([loc.point[1] for loc in locations if loc.classification == i])
        data_points_y = np.array([loc.point[0] for loc in locations if loc.classification == i])
        plt.plot(data_points_x, data_points_y, '.', color=colors[i], alpha=0.75)
        if centers is not None:
            plt.plot(centers[i][1], centers[i][0], 'bX', markersize=20)
        
    #Plot unclassified points:
    data_points_x = np.array([loc.point[1] for loc in locations if loc.classification is None])
    data_points_y = np.array([loc.point[0] for loc in locations if loc.classification is None])
    plt.plot(data_points_x, data_points_y, '.', alpha=0.75)

    plt.savefig(img_file_name, bbox_inches='tight', dpi=200)


def print_options(verbose, csv_path, k, plot, lim):
    print("Options:")
    print(f"    Verbose = {verbose}")
    print(f"    Input file = {csv_path}")
    print(f"    k = {k}")
    print(f"    Output plot = {plot}")
    print(f"    Distance limit = {lim}")
    print("")


def print_usage():
    print("\nUsage:")
    print("    python3 clustering_fun.py -k={clusters} [options] [/path/to/input/file.csv]")

    print("\nOptions:")
    print("    -k={value} (Required, number of clusters 1 to 1000)")
    print("    -n={value} (Number of points to gaussian generate if no CSV file is provided)")
    print("    -l={value} (Optional limit on distance from center to a location being classified in km)")
    print("    -p (plot output)")
    print("    -v (verbose output)")

    print("\nIf a csv file is not provided, the number of points must be specified with -n={value}")
    print("If a csv file is provided then the -n option is ignored")


if __name__ == '__main__':
    main()