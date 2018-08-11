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
        self.section = 'vehicle'
        self.con['homogeneous'] = True

    def vehicle(self):
        # Vehicle parameters
        self.__checkHomo()
        # This flag show indicates if the vehicle shall be considered as having
        # the same values of structural mass and thrust for all stages
        if self.con['homogeneous']:
            self.__getVehicleHomogeneous(self.config)
        else:
            self.__getVehicleHeterogeneous(self.config)

    def __checkHomo(self):
        #  The vehicle type is defined by the number of imputs on the variables
        #  T, eff and Isp.
        #  The options are:
        #  All with just one input
        #  All with a equal number of inputs diferent of one
        NStag = self.config.getint(self.section, 'NStag')

        auxstr = self.config.get(self.section, 'T')
        auxstr = auxstr.split(',')
        nT = len(auxstr)

        auxstr = self.config.get(self.section, 'efes')
        auxstr = auxstr.split(',')
        nEff = len(auxstr)

        auxstr = self.config.get(self.section, 'Isp')
        auxstr = auxstr.split(',')
        nIsp = len(auxstr)

        if nT != nEff or nT != nIsp:
            raise Exception('itsme saying: problems in vehicle configuration!')

        if nT > 0:
            self.con['homogeneous'] = False

        if nT > 1 and nT != NStag:
            raise Exception('itsme saying: NStag not coherent')

    def __getVehicleHomogeneous(self, config):

        items = config.items(self.section)

        for para in items:
            self.con[para[0]] = config.getfloat(self.section, para[0])

        # Number of stages
        self.con['NStag'] = config.getint(self.section, 'NStag')
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

    def __getVehicleHeterogeneous(self, config):

        section = 'vehicle'
        items = config.items(section)

        for para in items:
            if (para[0] != 'efes') and (para[0] != 'T') and (para[0] != 'Isp'):
                self.con[para[0]] = config.getfloat(section, para[0])

        self.con['NStag'] = config.getint(section, 'NStag')  # Number of stages

        auxstr = config.get(section, 'Isp')
        # print(auxstr)
        # print(self.con['NStag'])
        self.con['Isplist'] = self.__getListString(auxstr, self.con['NStag'])

        auxstr = config.get(section, 'efes')
        self.con['efflist'] = self.__getListString(auxstr, self.con['NStag'])

        auxstr = config.get(section, 'T')
        self.con['Tlist'] = self.__getListString(auxstr, self.con['NStag'])

    def __getListString(self, auxstr: str, Nstag: int) -> list:
        #  This method recognizes if just one imput was inserted in a variable.
        #  In this case, it generates a homogeneous list.
        auxstr = auxstr.split(',')
        auxnum = []
        if len(auxstr) == 1 and Nstag > 1:
            nn = float(auxstr[0])
            for ii in range(0, Nstag):
                auxnum.append(nn)

        else:
            for n in auxstr:
                auxnum.append(float(n))

        return auxnum


if __name__ == "__main__":

    con = dict()
    con['itsFile'] = "testHet.its"
    #  con['itsFile'] = "default3st.its"
    aux = modelConfiguration(con)
    aux.vehicle()
    print(aux.con)
