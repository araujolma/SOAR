#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:31:15 2020

@author: levi
"""
import sys, datetime, shutil, os, traceback, numpy, ast
from configparser import ConfigParser
from interf import logger
from main import main

line = "#" * 66
b = 'BATCH: '

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
        # TODO: modify the logger object to allow for message "headers" like BATCH
        self.log.printL(b+'Running a batch specified in:\n'+self.confFile+'\n')
        self.confFile = shutil.copy2(self.confFile, self.log.folderName +
                                     os.sep)
        self.log.printL(b+'Saved a copy of the configuration file to:\n'+
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

        elif self.mode == 'variations':
            sec = 'variations_mode'
            #self.NCases = Pars.getint(sec, 'NCases')
            self.probName = Pars.get(sec, 'probName')
            self.baseFile = Pars.get(sec, 'baseFile')
            # TODO: actually use this somewhere. It makes a lot of sense for variations
            self.initGuesMode = Pars.get(sec, 'initGuesMode')

            # Get the variations; NCases goes automatically
            varsSets_list = Pars.get(sec, 'vars').split(',\n')
            varSets = self.parseVars(varsSets_list)
            self.varSets = varSets
            self.NCases = len(varSets)+1

            # Generate the files corresponding to the cases to be run:
            self.genFiles()

        self.log.printL("\nThese are the parameters for the batch:")
        self.log.pprint(self.__dict__)


    def parseVars(self,varsSets_list):
        """Parse the variations to be performed, from the input file.
        The idea is that the user can put as many variations as desired in each run.
        For example, the input:

        vars = 'sgra','GSS_PLimCte',1.0e5  | 'accel','acc_max',100 | 'restr','h',2. ,
               'sgra','GSS_PLimCte',1.0e10 | 'accel','acc_max',100

        produces two runs on top of the base run,
        - on the 1st: 'GSS_PLimCte' parameter in 'sgra' section is changed to 1e5;
                      'acc_max' parameter in 'accel' section is changed to 100;
                      'h' parameter in 'restr' section is changed to 2.
        - on the 2nd: 'GSS_PLimCte' parameter in 'sgra' section is changed to 1e10;
                      'acc_max' parameter in 'accel' section is changed to 100.

        """
        self.log.printL("\n" + b + "Parsing the variations:")
        contVarSets = 0; varSets = []
        # for each set of variations (i.e., for each new run)
        for thisVarSet_str in varsSets_list:
            contVarSets += 1
            # display this set of variations, by number
            self.log.printL("\n\nThis is the variations set #{}:".format(contVarSets))
            # get the list by splitting into pieces separated by "|"
            thisVarSet_list = thisVarSet_str.split(' | ')
            contVars = 0; thisVarSet = []
            # for each individual variation:
            for var_Str in thisVarSet_list:
                contVars += 1
                # display variation by number
                self.log.printL("\n- This is the variation #{}:".format(contVars))
                thisVar = var_Str.split(',')
                msg = "\nSection: {}\nParam name: {}\nValue (as string): " \
                      "{}".format(thisVar[0], thisVar[1], thisVar[2])
                self.log.printL(msg)
                # assemble dictionary for easy referencing later
                var = {'section': thisVar[0].strip(),
                       'parName': thisVar[1].strip(),
                       'val': thisVar[2].strip()}
                thisVarSet.append(var)
            varSets.append(thisVarSet)

        self.log.printL("\n" + b + "This is the parsed varSets:")
        self.log.pprint(varSets)
        return varSets

    def genFiles(self):
        # generate the different files here, load their names to BM.baseFileList
        pass

if __name__ == "__main__":
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

    for runNr in range(BM.NCases):
        thisProb, thisFile = BM.probList[runNr], BM.baseFileList[runNr]
        BM.log.printL('\n'+line)
        msg = '\nBATCH: Running case {} of {}:' \
              '\n       Problem: {}, file: {}\n'.format(runNr+1,BM.NCases,thisProb,thisFile)
        BM.log.printL(msg)
        BM.log.printL(line+'\n')
        # set up the string for this run's number
        # (zero padded for keeping alphabetical order in case of more than 9 runs)
        runNrStr = str(runNr + 1).zfill(int(numpy.floor(numpy.log(BM.NCases))))
        # set up the this run's folder, inside the batch folder
        folder = BM.log.folderName + os.sep + runNrStr + '_'

        try:
            main(('',thisProb,thisFile), isManu=False, destFold=folder)
            BM.log.printL("\nBATCH: This run was completed successfully.")
        except KeyboardInterrupt:
            BM.log.printL("\nBATCH: User has stopped the program.")
        except Exception:
            BM.log.printL('\nBATCH: Sorry, there was something wrong with this run:')
            BM.log.printL(traceback.format_exc())

    BM.log.printL("\nBATCH: Execution finished. Terminating now.\n")
    BM.log.close()
