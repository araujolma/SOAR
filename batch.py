#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:31:15 2020

@author: levi
"""
import sys, datetime, shutil, os
from interf import logger
from configparser import ConfigParser

class batMan:
    """This is a class for the manager of batch runs, the BATch MANager."""

    def __init__(self,confFile=''):
        self.confFile = confFile

        Pars = ConfigParser()
        Pars.optionxform = str
        Pars.read(confFile)

        # load main parameters
        sec = 'main'
        self.name = Pars.get(sec, 'name')
        self.mode = Pars.get(sec, 'mode')
        self.logMode = Pars.get(sec, 'logMode')

        # create a logger for the batch
        self.log = logger('batch', runName=self.name, mode=self.logMode)
        self.log.printL('Running a batch specified in:\n'+self.confFile+'\n')
        self.confFile = shutil.copy2(self.confFile, self.log.folderName +
                                     os.sep)
        self.log.printL('Saved a copy of the configuration file to:\n'+
                        self.confFile+'\n')

        if self.mode == 'explicit':
            sec = 'explicit_mode'
            self.NCases = Pars.getint(sec,'NCases')
            this_str = Pars.get(sec, 'probs')
            self.probList = this_str.split(',\n')
            this_str = Pars.get(sec, 'baseFiles')
            self.baseFileList = this_str.split(',\n')

            # TODO: put a simple test for singleton lists, then replace them for proper
            #  lists of the same element

        self.log.printL("\nThese are the parameters for the batch:")
        self.log.pprint(self.__dict__)


if __name__ == "__main__":
    line = "#" * 66
    print('\n'+line)
    print('\nRunning batch.py with arguments:')
    print(sys.argv)
    print(datetime.datetime.now())
    args = sys.argv
    if len(args) == 1:
        # change this line to run this from the editor
        confFile = 'defaults' + os.sep + 'testAll.bat'
    else:
        # if the user runs the program from the command line,
        confFile = args[1]

    BM = batMan(confFile=confFile)

    try:
        pass
    # Manage the exceptions.
    # Basically, the logger must be safely shut down.
    except KeyboardInterrupt:
        BM.log.printL("\n\n\nUser has stopped the program.")
        raise
    except:
        BM.log.printL("\n\n\nI'm sorry, something bad happened.")
        raise
    finally:
        BM.log.printL("\nTerminating now.\n")
        BM.log.close()
