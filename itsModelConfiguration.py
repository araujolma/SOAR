#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:54:18 2018

@author: carlos
"""

import configparser


class modelConfiguration():

    def __init__(self, con: dict):
        # TODO: solve the codification problem on configuration files
        self.con = con
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(self.con['itsFile'])

    def vehicle(self):
        # Vehicle parameters
        section = 'vehicle'

        if not self.config.has_option(section, 'homogeneous'):
            self.con['homogeneous'] = True
        else:
            self.con['homogeneous'] = \
                self.config.getboolean(section, 'homogeneous')

        # This flag show indicates if the vehicle shall be considered as having
        # the same values of structural mass and thrust for all stages
        if self.con['homogeneous']:
            self.__getVehicleHomogeneous(self.config)
        else:
            self.__getVehicleHeterogeneous(self.config)

    def __getVehicleHomogeneous(self, config):

        section = 'vehicle'
        items = config.items(section)

        for para in items:
            self.con[para[0]] = config.getfloat(section, para[0])

        # Number of stages
        self.con['NStag'] = config.getint('vehicle', 'NStag')
        self.con['Isp1'] = self.con['Isp']
        self.con['Isp2'] = self.con['Isp']

        # This flag show indicates if the vehicle shall be considered as having
        # the same
        # values of structural mass and thrust for all stages
        efflist = []
        Tlist = []
        if self.con['NStag'] > 1:
            for jj in range(0, self.con['NStag']):
                efflist = efflist+[self.con['efes']]
                Tlist = Tlist+[self.con['T']]
        else:
            # This cases are similar to NStag == 2,  the differences are:
            # for NStag == 0 no mass is jetsoned
            # for NStag == 1 all structural mass is jetsoned at the end of all
            # burning
            for jj in range(0, 2):
                efflist = efflist+[self.con['efes']]
                Tlist = Tlist+[self.con['T']]

        self.con['efflist'] = efflist
        self.con['Tlist'] = Tlist
