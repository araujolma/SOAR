#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:31:15 2020

@author: levi
"""
import sys, datetime, shutil, os, traceback, numpy, random, string, time
import sgra
from configparser import ConfigParser, NoOptionError
from interf import logger
from main import main

line = "#" * 66
b = 'BATCH: '
nb = '\n'+b

class batMan:
    """This is a class for the manager of batch runs, the BATch MANager."""

    def __init__(self, confFile=''):
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

        try:
            # this is a list of the attributes for the "sol" object that the user wants
            cmdList = Pars.get(sec,'postProcKeyList').split(',\n')
            # assemble the list of keys (basically clean parentheses, brackets, etc)
            keyList = []
            for i in range(len(cmdList)):
                cmd = cmdList[i].strip()
                key = cmd + ''
                # brackets get mapped to _ ; quotes get mapped to nothing
                key = key.replace('[','_').replace('(','_').replace('{','_').\
                    replace('.','_').replace(']','').replace(')','').replace('}','').\
                    replace('"','').replace("'","")
                keyList.append(key)
                cmdList[i] = cmd
            self.postProcKeyList = keyList
            self.postProcCmdList = cmdList
        except NoOptionError:
            # apparently the user failed to specify what he/she wants. Default it is!
            self.log.printL(nb+"Could not find 'postProcKeyList' in the .bat file."
                               "\n  Going with default keys...")
            self.postProcKeyList = []
            self.postProcCmdList = []
        for key in ['NIterGrad', 'GSStotObjEval', 'GSSavgObjEval', 'timer']:
            self.postProcKeyList.append(key)
            self.postProcCmdList.append(key)

        if self.mode == 'explicit':
            sec = 'explicit_mode'
            self.NCases = Pars.getint(sec,'NCases')
            this_str = Pars.get(sec, 'probs')
            self.probList = this_str.split(',\n')
            # If the problem list is too short, repeat the last element
            if len(self.probList) < self.NCases:
                self.probList = self.repLastElem(self.probList,
                                                 self.NCases-len(self.probList))
            this_str = Pars.get(sec, 'baseFiles')
            self.baseFileList = this_str.split(',\n')
            # If the problem list is too short, repeat the last element
            if len(self.baseFileList) < self.NCases:
                self.baseFileList = self.repLastElem(self.baseFileList,
                                                 self.NCases-len(self.baseFileList))

        elif self.mode == 'variations':
            sec = 'variations_mode'
            self.baseFile = Pars.get(sec, 'baseFile')
            self.initGuesMode = Pars.get(sec, 'initGuesMode')

            # Get the variations; NCases goes automatically
            varsSets_list = Pars.get(sec, 'vars').split(',\n')
            varSets = self.parseVars(varsSets_list)
            self.NCases = len(varSets)+1

            # List of the problems
            self.probList = [Pars.get(sec, 'probName').strip()] * self.NCases

            # Special procedures for initGuesMode 'base' or 'cascade'
            if self.initGuesMode == 'base':
                # For initial guess in 'base mode', add these variations
                base_sol_loc = self.folderNameForRun(0) + self.probList[0] + os.sep
                d1 = {'section': 'settings',
                       'parName': 'defOpt',
                       'val': 'loadSol'}
                d2 = {'section': 'settings',
                       'parName': 'loadSolDir',
                       'val': base_sol_loc + 'finalSol.pkl'}
                for i in range(self.NCases-1):
                    varSets[i].append(d1)
                    varSets[i].append(d2)
            elif self.initGuesMode == 'cascade':
                # For initial guess in 'cascade mode', add these variations
                d1 = {'section': 'settings',
                       'parName': 'defOpt',
                       'val': 'loadSol'}
                for i in range(self.NCases-1):
                    base_sol_loc = self.folderNameForRun(i) + self.probList[i] + \
                                   os.sep
                    d2 = {'section': 'settings',
                          'parName': 'loadSolDir',
                          'val': base_sol_loc + 'finalSol.pkl'}
                    varSets[i].append(d1)
                    varSets[i].append(d2)

            self.varSets = varSets

            # Generate the files corresponding to the cases to be run:
            self.genFiles()
        # Reference for successful run
        self.isGoodRun = numpy.zeros(self.NCases, dtype=numpy.bool_)

        # post processing info
        self.postProcInfo = {}

        # basic initialization (if something goes wrong, the field stays as 'None')
        for key in self.postProcKeyList:
            self.postProcInfo[key] = [None] * self.NCases

        # show parameters and leave
        self.log.printL("\nThese are the parameters for the batch:")
        self.log.pprint(self.__dict__)

    @staticmethod
    def repLastElem(givnList: list, Nrep: int):
        """Repeat the last element of the given list, for a given number of times."""
        elem = givnList[-1]
        for i in range(Nrep):
            givnList.append(elem)
        return givnList

    def runNumbAsStr(self, runNr: int):
        """Produce the standard run number, starting from 1, zero padded for keeping
        alphabetical order in case of more than 9 runs."""

        return str(runNr + 1).zfill(int(numpy.floor(numpy.log(self.NCases))))

    def folderNameForRun(self, runNr: int):
        """Procude the folder name for a given run number."""

        # set up the string for this run's number
        runNrStr = self.runNumbAsStr(runNr)
        # the name of this run's folder, inside the batch folder
        folder = self.log.folderName + os.sep + runNrStr + '_'

        return folder

    def parseVars(self, varsSets_list):
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
        self.log.printL(nb + "Parsing the variations:")
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

        return varSets

    def genFiles(self):
        """Generate the different files, loading their names to self.baseFileList."""

        self.log.printL("\n"+b+"Generating the files for the variations...")

        l = len(self.baseFile)

        # generate a random code for minimizing the chance of file conflict
        # TODO: actually checking for conflicts should not be that hard!
        rand = ''.join(random.choice(string.ascii_letters + string.digits)
                       for _ in range(10))

        # base file list starts with the first case, which is the base file
        baseFileList = [self.baseFile]

        # one file for each of the runs, except the first
        for nRun in range(self.NCases-1):
            name = self.baseFile[:l-4] + '_' + rand + \
                   '_var' + self.runNumbAsStr(nRun) + '.its'
            self.log.printL("Name of the file for run #{}: {}".format(nRun+1, name))
            # Append the name of the file to the list of base files, for running later!
            baseFileList.append(name)
            # List of Variations for this run
            theseVars = self.varSets[nRun]

            # List (actually, set) of sections
            secList = set()
            for var in theseVars:
                secList.add(var['section'])

            # open the file pointers
            targHand = open(name, 'w+')          # target file
            baseHand = open(self.baseFile, 'r')  # source file

            secName = ''; newLine = ''
            for line in baseHand:
                # remove spaces, \n and other funny characters
                stripLine = line.strip()

                # check if this line has to be changed (implement variation) or not
                mustChange = False
                # get current section name (if it changes)
                if stripLine.startswith('['):
                    # new section found, maybe?
                    ind = stripLine.find(']')
                    if ind > -1:
                        # closing bracket found; new section confirmed
                        secName = stripLine[1:ind]
                else:
                    # not a new section. Maybe a comment!
                    if stripLine.startswith('#'):
                        continue
                    # Not a comment... Now it has to be a variable assignment.

                    # if this section is not even in the list, no need for checking
                    # get the equal sign for separating variable name
                    if secName not in secList:
                        targHand.write(stripLine + '\n')
                        continue

                    # variable name goes right up to the =
                    ind = stripLine.find('=')
                    if ind == -1:
                        targHand.write(stripLine + '\n')
                        continue

                    varbName = stripLine[:ind].strip()
                    # check for matches: the section must be equal to the current
                    # section and the variable name must match the 'parName'
                    for var in theseVars:
                        # DISCLAIMER:
                        # if the user is crazy enough so that there are two or
                        # more variations on the same set for the same variable,
                        # only the last one will be applied.

                        if var['section'] == secName and var['parName'] == varbName:

                            # Finally! Change the line
                            mustChange = True
                            newLine = stripLine[:ind+1] + ' ' + var['val']

                # finally, write either the original line or the changed line
                if mustChange:
                    targHand.write(newLine + '\n')
                else:
                    targHand.write(stripLine + '\n')
            # close file handlers
            targHand.close()
            baseHand.close()

        self.baseFileList = baseFileList

    # noinspection PyUnusedLocal
    def getPostProcData(self, solu: sgra.sgra, run_nr:int):
        """Post processing data gathering.

        This basically consists in retrieving fields from the solution object."""

        for key, cmd in zip(self.postProcKeyList, self.postProcCmdList):
            # try...except so that if something goes wrong,
            # the whole batch is not wasted
            try:
                self.postProcInfo[key][run_nr] = eval("solu.{}".format(cmd))
            except AttributeError:
                msg = "Error while retrieving the attribute" \
                      " {} (command: '{}') in run {}".format(key, cmd, run_nr)
                self.log.printL(nb + msg)
                self.postProcInfo[key][run_nr] = 'ERROR!'

    def showPostProcData(self):
        """Post processing data show"""
        self.log.printL(nb + "Post-processing results:")
        self.log.pprint(BM.postProcInfo)

        try:
            import xlsxwriter
            # declare workbook and worksheet
            workbook = xlsxwriter.Workbook(self.log.folderName + os.sep +
                                           'Results_' + self.log.folderName +'.xlsx')
            ws = workbook.add_worksheet()
            # heading
            ws.write(0, 0, 'Run #')
            ws.write(0, 1, 'Problem name')
            ws.write(0, 2, 'Base file')
            ws.write(0, 3, 'Good run?')
            col = 4
            # write the key names
            for key in self.postProcKeyList:
                ws.write(0,col,key)
                col += 1
            # fill in the actual data
            for runNr in range(self.NCases):
                row = runNr + 1 # because the first row has the header
                ws.write(row, 0, runNr + 1)
                ws.write(row, 1, self.probList[runNr])
                ws.write(row, 2, self.baseFileList[runNr])
                ws.write(row, 3, self.isGoodRun[runNr])
                col = 4
                for key in self.postProcKeyList:
                    ws.write(row, col, self.postProcInfo[key][runNr])
                    col += 1
            workbook.close()

        except ImportError:
            self.log.printL(nb+"Error while importing xlsxwriter...")

            # this is an attempt to print out the data
            titl = "\nRun #\tProbName\tBaseFile\tGood run?"
            for key in self.postProcKeyList:
                titl += '\t'+key
            self.log.printL(titl+'\n')
            for runNr in range(self.NCases):
                msg = '{}\t{}\t{}\t{}'.format(runNr,self.probList[runNr],
                                              self.baseFileList[runNr],
                                              self.isGoodRun[runNr])
                for key in self.postProcKeyList:
                    msg += '\t{}'.format(self.postProcInfo[key][runNr])
                self.log.printL(msg)

    def countdown(self,sec=3):
        self.log.printL("\nBATCH: Starting in...")
        for i in range(sec,0,-1):
            self.log.printL("{}...".format(i))
            time.sleep(1.)

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
        msg = nb + 'Running case {} of {}:' \
              '\n       Problem: {}, file: {}\n'.format(runNr+1,BM.NCases,thisProb,
                                                        thisFile)
        BM.log.printL(msg)
        BM.log.printL(line+'\n')

        # the name of this run's folder, inside the batch folder
        folder = BM.folderNameForRun(runNr)

        try:
            BM.countdown()
            sol, solInit = main(('',thisProb,thisFile), isManu=False, destFold=folder)
            BM.log.printL(nb + "This run was completed successfully.")
            BM.log.printL(nb + "Entering post-processing for this run...")
            BM.getPostProcData(sol,runNr)
            BM.log.printL(nb + "Done. Going for the next run.")
            BM.isGoodRun[runNr] = sol.isConverged  # good or bad run
        except KeyboardInterrupt:
            BM.log.printL(nb + "User has stopped the program during this run.")
        except Exception:
            BM.log.printL(nb + 'Sorry, there was something wrong with this run:')
            BM.log.printL(traceback.format_exc())
        finally:
            if BM.mode == 'variations':
                # clean up the generated .its files
                if runNr > 0:
                    file = BM.baseFileList[runNr]
                    BM.log.printL(nb + "Removing file: " + file)
                    os.remove(file)

    if BM.isGoodRun.all():
        BM.log.printL(nb+"* * * Every run was successful! * * *")
    else:
        BM.log.printL(nb + "* * * WARNING!! NOT ALL RUNS WERE SUCCESSFUL!!! * * *")
    BM.showPostProcData()
    BM.log.printL(nb+"Execution finished. Terminating now.\n")
    BM.log.close()
