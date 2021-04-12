#!/usr/bin/env python3

import sys
import csv
import random
import math
import location
import numpy as np
import matplotlib.pyplot as plt 
from operator import itemgetter
from haversine import haversine, Unit

class KMeans():
    def __init__(self, data, k, verbose, dist_limit=None):
        self.data = data
        self.k = k
        self.verbose = verbose
        self.distance_limit = dist_limit


    def classify(self, start_centers):
        print(f"\nClassifying with k={self.k}:")

        centers = start_centers
        iteration = 0
        not_converged = True

        while (not_converged):
            self._iterate_classify(centers)
            means = self._find_means(centers)
            iter_variance = self._sum_variance()
            if self.verbose:
                print(f"Converging iteration {iteration}, variance = {iter_variance} km")

            iteration += 1
            not_converged = self._means_differ(means, centers)
            centers = means

        if self.verbose:
            print(f"\nConverged after {iteration} iterations")
            print(f"Total classification variance = {iter_variance} km")
            
        return centers, self.data


    def init_centers(self):
        centers = np.array([random.sample(self.data, len(self.data))[0].point])

        while len(centers) < self.k:
            dist_sq = self._get_distance_from_centers(centers)
            centers = np.append(centers, [self._choose_next_center(centers, dist_sq)], axis=0)

        return centers, sum(self._get_distance_from_centers(centers))


    def _choose_next_center(self, centers, dist_sq):
        probs = dist_sq / dist_sq.sum()
        sum_probs = probs.cumsum()

        r = random.random()
        idx = np.where(sum_probs >= r)[0][0]
        return self.data[idx].point


    def _get_distance_from_centers(self, centers):
        return np.array([min([haversine(d.point, c) for c in centers]) for d in self.data])


    def _iterate_classify(self, centers):
        for d in self.data:
            distances = np.array([haversine(d.point, c) for c in centers])
            # Minimum in the enumerated list (value is item index 1)
            idx, dist = min(enumerate(distances), key=itemgetter(1))
            if self.distance_limit is not None and dist > self.distance_limit:
                d.classification = None
                d.class_distance = None
            else:
                d.classification = idx
                d.class_distance = dist


    def _find_means(self, centers):
        means = np.array([
            [
                np.mean([l.point[0] for l in self.data if l.classification == classification]),
                np.degrees(np.arctan2(
                    np.mean([np.sin(np.radians(l.point[1])) for l in self.data if l.classification == classification]),
                    np.mean([np.cos(np.radians(l.point[1])) for l in self.data if l.classification == classification])
                    
                ))
            ]
            for (classification, center) in enumerate(centers)])
        return means


    def _means_differ(self, means, points):
        for mean, point in zip(means, points):
            if not np.array_equal(mean, point):
                return True

        return False


    def _sum_variance(self):
        return sum([d.class_distance for d in self.data if d.class_distance is not None])