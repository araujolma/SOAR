#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:27:45 2020

@author: levi

A simple module for calculating the differences between cases.
This is specially useful for trying to figure out what was being done when some cases
were tried.

This modules is intended to be run from the command line, like this:

$ python calc_diff_cases.py
or
$ python calc_diff_cases.py 2020-09-01
or
$ python calc_diff_cases.py 2020-09-01 2020-10-12

The first run calculates the difference between successive run cases for all cases
on the current directory.
The second run does the same as the first, but filters out cases before september 1st,
2020.
The third run does the same as the first, but filters out cases before september 1st,
2020 and after october 12th, 2020.
"""

import os
import datetime
import sys

def find_its(folder):
    """Find a .its file in a given folder."""

    for file in os.listdir(folder):
        if file.endswith(".its"):
            # found it!
            case = folder + os.sep + file
            return case
    else:
        print("No .its found in {}.".format(folder))
        return None

def calc_diff(case1, case2, num1=None, num2=None):
    """Calculate the difference between given files."""

    if case1 is None:
        print("No file specified in case1. Skipping...")
        return None
    if case2 is None:
        print("No file specified in case2. Skipping...")
        return None

    if num1 is None:
        print("\nDifferences between {}".format(case1))
    else:
        print("\nDifferences between ({}) {}".format(num1, case1))
    if num2 is None:
        print("{}and {}:".format(' ' * 16, case2))
    else:
        print("{}and ({}) {}:".format(' ' * 16, num2, case2))

    # Properly calculate the differences
    str_cmd = 'diff {} {}'.format(case1, case2)
    stream = os.popen(str_cmd)
    output = stream.read()
    print(output)

def compare_base(base_case: str = None, run_dir_list: list = None):
    """Compare a given case (base) with a list of cases."""

    if base_case is None:
        base_case = 'probLand_2020_11_21_14_02_00_002344' + os.sep + \
                    'probLand-treta.its'
    if run_dir_list is None:
        run_dir_list = ['probLand_2020_09_20_12_11_07_983239',
                        'probLand_2020_09_20_12_15_46_179188',
                        'probLand_2020_09_20_12_22_01_204241',
                        'probLand_2020_09_20_12_32_47_202888',
                        'probLand_2020_09_20_12_41_51_475844',
                        'probLand_2020_09_20_13_05_01_696468',
                        'probLand_2020_09_20_19_48_39_359390',
                        'probLand_2020_09_29_19_07_41_999247']

    for run_dir in run_dir_list:
        case = find_its(run_dir)
        calc_diff(base_case, case)

def successive_vars(run_dir_list:list = None):
    """Calculate successive variations between files."""
    if run_dir_list is None:
        run_dir_list = ['probLand_2020_09_20_12_11_07_983239',
                        'probLand_2020_09_20_12_15_46_179188',
                        'probLand_2020_09_20_12_22_01_204241',
                        'probLand_2020_09_20_12_32_47_202888',
                        'probLand_2020_09_20_12_41_51_475844',
                        'probLand_2020_09_20_13_05_01_696468',
                        'probLand_2020_09_20_19_48_39_359390',
                        'probLand_2020_09_29_19_07_41_999247']

    base_case = find_its(run_dir_list[0])
    num1 = 1
    for i in range(1,len(run_dir_list)):
        num2 = i+1
        case = find_its(run_dir_list[i])
        calc_diff(base_case, case, num1=num1, num2=num2)
        # current case becomes base case
        base_case = case
        num1 = num2

def get_case_folders(since=None, until=None):
    """Get the folders corresponding to the cases of interest."""
    if since is None:
        since = datetime.datetime(2016, 2, 19, 12, 0, 0, 0).isoformat()
    if until is None:
        until = datetime.datetime.now().isoformat()

    # assemble a dictionary with the runs
    runs = dict()
    for ind, file in enumerate(os.listdir()):
        if os.path.isdir(file) and file.startswith('prob'):
            flist = file.split('_')
            ilist = [int(flist[i]) for i in range(1,8)]
            this_date = datetime.datetime(*ilist)
            this_date = this_date.isoformat()
            if since < this_date < until:
                runs[this_date] = file

    # assemble list from dictionary, from the first case to the last, chronologically
    run_list = list()
    print("\nThese are the cases of interest:")
    n = 0
    for key in sorted(runs.keys()):
        n += 1
        run_list.append(runs[key])
        print("  ({}) {}".format(n,runs[key]))

    return run_list


if __name__ == "__main__":
    #compare_base()
    #successive_vars()

    narg = len(sys.argv)
    if narg > 1:
        if narg > 2:
            run_dir_list = get_case_folders(sys.argv[1], sys.argv[2])
        else:
            run_dir_list = get_case_folders(sys.argv[1])
    else:
        run_dir_list = get_case_folders()

    successive_vars(run_dir_list)