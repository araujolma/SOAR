#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:35:29 2017

@author: levi
"""

import dill, datetime, pprint, os, shutil, time, xlsxwriter, sgra
from utils import getNowStr
from configparser import ConfigParser

bscImpStr = "\n >> "
dashStr = '\n' + '-' * 88

class logger:
    """ Class for the handler of log messages."""

    def __init__(self,probName, isManu=True, makeDir=True, path='', runName='',
                 mode='both', dateTag=True, xlsx=False):
        """
        :param probName: Problem name (for a proper folder name);
        :param isManu: flag for determinating if in manual mode or not;
        :param makeDir: flag for making the directory or not.
                        It overrides all following parameters;
        :param path: Allows user specification for where the folder should be
        :param runName: further customization for folder name, seldom used
        :param mode: logging output. 'screen' prints to screen, 'file' prints to file
                     only, and 'both' does both;
        :param dateTag: flag for putting or not a date tag at the end of the folder.
        """


        # Mode ('both', 'file' or 'screen') sets the target output.
        # This can be overriden in each .printL() call, though

        if mode.startswith('both'):
            self.mode = 0
        elif mode.startswith('screen'):
            self.mode = 1
        elif mode.startswith('file'):
            self.mode = 2
        else:
            msg = "Unknown logger mode: '{}'. " \
                  "Proceeding with 'screen' mode.".format(mode)
            print(msg)
            self.mode = 1

        # this determines the necessity of manual input:
        self.isManu = isManu

        # Run status (for repoStat), by default it begins with 'none'
        self.runStatRep = 'none'

        self.xlsx = None
        if makeDir:  # BE CAREFUL WHEN CHANGING THIS!! IT WILL BREAK BATCH.PY a.s. !!!

            # Results folder for this run
            self.folderName = path + probName
            # Append run name, if it is the case
            if len(runName) > 0:
                self.folderName += '_' + runName
            # put the date on the folder to avoid mix-ups with other runs
            if dateTag:
                self.folderName += '_' + getNowStr()
            # Create the folder
            os.makedirs(self.folderName)

            try:
                fullName = self.folderName + os.sep + 'log.txt'
                self.fhand = open(fullName, 'w+')
                self.printL("Creating a log in file: " + fullName)
                if xlsx:
                    # MS Excel does not allow one to open two spreadsheets with the
                    # same name, even if they have different paths. So, it was decided
                    # to replicate the folder name in the .xlsx file as well. Extra
                    # care must be taken with the os.sep character.
                    # Thanks, Microsoft!
                    name = self.folderName + os.sep + 'GradInfo_' + \
                           self.folderName.replace(os.sep,'_') + '.xlsx'
                    self.printL("\nCreating a spreadsheet in file: " + name)
                    workbook = xlsxwriter.Workbook(name)
                    step_tab = workbook.add_worksheet('step')
                    obj_tab = workbook.add_worksheet('obj')
                    P_tab = workbook.add_worksheet('P')
                    I_tab = workbook.add_worksheet('I')
                    J_tab = workbook.add_worksheet('J')
                    misc_tab = workbook.add_worksheet('misc')

                    self.xlsx = {'workbook': workbook, 'maxEvals': 0,
                                 'step': step_tab, 'obj': obj_tab,
                                 'P': P_tab, 'I': I_tab, 'J': J_tab,
                                 'misc': misc_tab}

            except:
                print("Sorry, could not open/create the file!")
                raise
        else:
            self.mode = 1 # screen
            self.folderName = os.getcwd()

    # TODO: when grinding for performance, changing 'mode' to an integer would be
    #  significantly faster than checking strings every single time!!

    def repoStat(self,runStat):#,iterN=None):
        """ Report the status of the current run, as commanded.
        The report is simply the creation of an empty .txt file whose title reports
         the
        progress of the run. There is a simple mechanism to prevent creation of the
         same
        file again.
        :param runStat: the status to be reported.
            Either 'init' or 'inProg', 'ok', 'usrStop' or 'err'.
        :return: None
        """
        if runStat == self.runStatRep:
            # Nothing changed. The program should only get here if some programmer
            # called this method by mistake.
            self.printL("\nThis run's status remains the same. "
                        "Ignoring report command.")
        else:
            self.runStatRep = runStat #updating the reported status

            # Delete previous status file(s)
            if not(self.runStatRep == 'none'):
                currDir = os.getcwd() + os.sep + self.folderName
                listedDir = os.listdir(currDir)
                for f in listedDir:
                    if f.startswith('!runStatus-') and f.endswith('.txt'):
                        os.remove(currDir + os.sep + f)

            fname = self.folderName + os.sep + '!runStatus-'
            if runStat == 'init':
                fname += 'initializing.txt'
            elif runStat == 'inProg':
                fname += 'in_progress.txt'
            elif runStat == 'ok':
                fname += 'successful!.txt'
            elif runStat == 'usrStop':
                fname += 'stopped_by_user.txt'#_in_it#{}.txt'.format(iterN)
            elif runStat == 'err':
                fname += 'error!'#_in_it#{}.txt'.format(iterN)
            self.printL("\nReporting status with file '{}'".format(fname))
            fhand = open(fname,'w+') # create the file
            fhand.close() # close handler


    def printL(self,msg,mode=-1):
        """ Print a given message.
        Depending on the self.mode, it can be on screen or on a file or in both.

        :param msg: the message to be printed
        :param mode: override the mode of print.
        :return: None
        """

        if mode == -1:
            mode = self.mode

        if isinstance(mode,str):
            raise(Exception("logger: 'mode' cannot be used as string anymore. "
                            "I'm sorry."))

        if mode == 0:
            # both file and screen
            print(msg)
            self.fhand.write('\n'+msg)
        elif mode == 2:
            # file only
            self.fhand.write('\n'+msg)
        else:
            # screen only
            print(msg)

    def pprint(self,obj):
        """Pretty print."""

        if self.mode == 0:
            # both file and screen
            pprint.pprint(obj)
            pprint.pprint(obj, self.fhand)
        elif self.mode == 2:
            # file only
            pprint.pprint(obj,self.fhand)
        else:
            # screen only
            pprint.pprint(obj)

    def prntDashStr(self):
        self.printL(dashStr)

    def prom(self,msg=''):
        """Prompt the user for some input"""

        if msg not in '':
            self.printL(msg)

        if self.isManu:
            inp = input(bscImpStr) # the string 'bscImpStr' is printed on screen here
        else:
            # override the input if it is not on manual mode (e.g. batch)
            inp = ''

        # If the output options contain "file mode" print to file only, since the
        # screen was already covered by built-in function 'input'
        if self.mode == 0 or self.mode == 2: #
            self.printL(bscImpStr + inp, mode=2)
        return inp

    def close(self):
        """ Close the txt file and the spreadsheet if it is the case."""

        if self.xlsx is not None:
            # Writing the "legends" for most tabs
            for key in ['step', 'P', 'I', 'J', 'obj']:
                self.xlsx[key].write(0, 0, 'Base value')
                for row in range(1, self.xlsx['maxEvals']+2):
                    self.xlsx[key].write(row, 0, 'Eval #{}'.format(row))

            # Writing the "legends" for the 'misc' tab - step limits
            row, col = 0, 0
            for key in ['pi','PLim','Obj0','ObjNeg']:
                self.xlsx['misc'].write(row, col, key+'-StepLimActv')
                row += 1
                self.xlsx['misc'].write(row, col, key+'-StepLimLowr')
                row += 1
                self.xlsx['misc'].write(row, col, key+'-minStepVio')
                row += 2
            # Writing the "legends" for the 'misc' tab - trios
            self.xlsx['misc'].write(row, col, 'stepMinObj-lowrBnd')
            row += 1
            self.xlsx['misc'].write(row, col, 'stepMinObj')
            row += 1
            self.xlsx['misc'].write(row, col, 'stepMinObj-upprBnd')
            row += 2
            # Writing the "legends" for the 'misc' tab - stop motives
            self.xlsx['misc'].write(row, col, 'pLimSrchStopMotv')
            row += 1
            self.xlsx['misc'].write(row, col, 'stepSrchStopMotv')
            # Writing the "legends" for the 'misc' tab - P limit regression coefs
            row += 2
            self.xlsx['misc'].write(row, col, 'PLimRegrAngCoef')
            row += 1
            self.xlsx['misc'].write(row, col, 'PLimRegrLinCoef')

            # Close the spreadsheet
            #self.printL("Closing spreadsheet...")
            self.xlsx['workbook'].close()
            #self.printL("Spreadsheet closed.")
        #self.printL("Closing logger file...")
        self.fhand.close()

class ITman:
    """Class for the ITerations MANager.

    Only one object from this class is intended to be active during a program
    run. It manages the iterations in the solution, commands solution saving,
    plotting, debug modes, etc.

    """

    def __init__(self, confFile='', probName='prob', isInteractive=False, isManu=True,
                 destFold=''):
        """

        :param confFile: Configuration file name/path (.its file)
        :param probName: Problem name
        :param isInteractive: flag for interactive mode
        :param isManu: flag for manual mode (true unless being run in batch mode)
        :param destFold: destination folder path
        """

        # default values for overall settings; these will be overridden if
        # there is anything with the same name on the [settings] section of
        # the configuration file.
        self.probName = probName
        self.isManu = isManu
        self.isNewSol = True
        self.defOpt = 'newSol'#'loadSol'#
        self.initOpt = 'extSol'
        self.confFile = confFile
        self.loadSolDir = 'defaults' + os.sep + probName + '_solInitRest.pkl'
        self.loadAltSolDir = ''
        #'solInitRest.pkl'#'solInit.pkl'#'currSol.pkl'
        self.MaxIterGrad = 10000
        self.GRplotSolRate = 20#1#
        self.GRsaveSolRate = 100
        self.GRpausRate = 10000#1000#10
        self.GradHistShowRate = 20
        self.RestPlotSolRate = 20
        self.RestHistShowRate = 100#20
        self.ShowEigRate = 100
        self.ShowGRrateRate = 20
        # TODO: Are these rates too large? Add these parameters to the .its file
        #  and change them there!
        self.ShowLambRate = 10000
        self.ShowVarRate = 10000
        self.plotResPRate = 10000
        self.plotResQRate = 10000
        self.parallelOpt = {'gradLMPBVP': True,
                            'restLMPBVP': True}
        self.default_dbugOptRest = {'pausRest': False,
                                    'pausCalcP': False,
                                    'plotP_int': False,
                                    'plotP_intZoom': False,
                                    'plotIntP_int': False,
                                    'plotSolMaxP': False,
                                    'plotRsidMaxP': False,
                                    'plotErr': False,
                                    'plotCorr': False,
                                    'plotCorrFin': False}
        flag = False
        self.default_dbugOptGrad = {'pausGrad': flag,#True,#
                                    'pausCalcQ': flag,#True,#
                                    'prntCalcStepGrad': True,#flag,#
                                    'plotCalcStepGrad': flag,#True,#
                                    'manuInptStepGrad': flag,
                                    'pausCalcStepGrad': flag,#True,#
                                    'xlsxCalcStepGrad': True,#flag,#
                                    'plotQx': flag,
                                    'plotQu': flag,
                                    'plotLam': flag,
                                    'plotQxZoom': flag,
                                    'plotQuZoom': flag,
                                    'plotQuComp': flag,
                                    'plotQuCompZoom': flag,
                                    'plotSolQxMax': flag,
                                    'plotSolQuMax': flag,
                                    'plotCorr': flag,
                                    'plotCorrFin': flag,
                                    'plotF': flag,#True,
                                    'plotFint': flag,
                                    'plotI': flag}

        if isInteractive:
            # screen only mode
            self.log = logger(probName,runName='interactive',makeDir=False)
        else:
            # Create directory for logs and stuff;
            # dateTag only if manual flag is true
            self.log = logger(probName, isManu=self.isManu, path=destFold,
                              dateTag=self.isManu,
                              xlsx=self.default_dbugOptGrad['xlsxCalcStepGrad'])

        if len(confFile) > 0:
            # Get the configurations in the file.
            Pars = ConfigParser()
            Pars.optionxform = str
            Pars.read(confFile)

            # TODO: The best way would be to iterate since the names of the
            #  fields are essentially the same as the ones in the file...
            sec = 'settings'
            if sec in Pars.sections():
                if 'defOpt' in Pars.options(sec):
                    self.defOpt = Pars.get(sec, 'defOpt')
                if 'initOpt' in Pars.options(sec):
                    self.initOpt = Pars.get(sec, 'initOpt')
                if 'loadSolDir' in Pars.options(sec):
                    self.loadSolDir = Pars.get(sec, 'loadSolDir')
                if 'MaxIterGrad' in Pars.options(sec):
                    self.MaxIterGrad = Pars.getint(sec, 'MaxIterGrad')
                if 'GRplotSolRate' in Pars.options(sec):
                    self.GRplotSolRate = Pars.getint(sec, 'GRplotSolRate')
                if 'GRsaveSolRate' in Pars.options(sec):
                    self.GRsaveSolRate = Pars.getint(sec, 'GRsaveSolRate')
                if 'GRpausRate' in Pars.options(sec):
                    self.GRpausRate = Pars.getint(sec, 'GRpausRate')
                if 'GradHistShowRate' in Pars.options(sec):
                    self.GradHistShowRate = Pars.getint(sec, 'GradHistShowRate')
                if 'RestPlotSolRate' in Pars.options(sec):
                    self.RestPlotSolRate = Pars.getint(sec, 'RestPlotSolRate')
                if 'ShowEigRate' in Pars.options(sec):
                    self.ShowEigRate = Pars.getint(sec, 'ShowEigRate')
                if 'ShowGRrateRate' in Pars.options(sec):
                    self.ShowGRrateRate = Pars.getint(sec, 'ShowGRrateRate')
                if 'ShowLambRate' in Pars.options(sec):
                    self.ShowLambRate = Pars.getint(sec, 'ShowLambRate')
                if 'ShowVarRate' in Pars.options(sec):
                    self.ShowVarRate = Pars.getint(sec, 'ShowVarRate')
                if 'PlotResPRate' in Pars.options(sec):
                    self.plotResPRate = Pars.getint(sec, 'PlotResPRate')
                if 'PlotResQRate' in Pars.options(sec):
                    self.plotResQRate = Pars.getint(sec, 'PlotResQRate')
                if 'PrllGradLMPBVP' in Pars.options(sec):
                    self.parallelOpt['gradLMPBVP'] = \
                        Pars.getboolean(sec, 'PrllGradLMPBVP')
                if 'PrllRestLMPBVP' in Pars.options(sec):
                    self.parallelOpt['restLMPBVP'] = \
                        Pars.getboolean(sec, 'PrllRestLMPBVP')
            else:
                msg = '\nWarning: no [settings] section on input file "' + \
                        confFile + '".\nProceeding with default values.'
                self.log.printL(msg)

        self.overrideParallel()

    def overrideParallel(self):
        """Override the parallel configurations in order to prevent entering
        into parallel mode in Windows systems. (issue #63, btw)"""

        if os.sep != '/':
            # windows systems!
            msg = "\n" + dashStr + \
                  "\nOverriding the parallel settings to False!" + \
                  dashStr + "\n\n"
            self.log.printL(msg)
            self.parallelOpt = {'gradLMPBVP':False,
                                'restLMPBVP':False}

    def printPars(self):
        self.log.printL("\nThese are the attributes for the" + \
                        " Iterations manager:\n")
        self.log.pprint(self.__dict__)

    def bell(self, lag=0.1, nRing=3):
        """ Make some sounds to warn the user. Only works in manual mode.
         :param lag: lag between successive rings, in seconds;
         :param nRing: number of rings."""

        if self.isManu:
            for k in range(nRing):
                print("\a")
                time.sleep(lag)

    def greet(self):
        """This is the first command to be run at the beginning of the
        MSGRA.

        The idea is to let the user choose whether he/she wants to load a
        previously prepared solution, or generate a new one.

        This function does not return anything, but at least these parameters
        must be set (or confirmed):
        - self.isNewSol (boolean): false only for loaded solutions
        - self.initOpt (str): 'extSol', 'default' or 'naive(2)', applicable
        only when isNewSol==True
        - self.loadSolDir (str): directory for loading solution, applicable
        only when isNewSol==False, of course
        - self.altSolDir (str):  directory for loading alternative solution,
        applicable only when isNewSol==False, of course
        """

        # First greetings to user
        self.log.prntDashStr()
        self.log.printL("\nWelcome to SGRA!\n")
        # Inform problem
        self.log.printL("Loading settings for problem: "+self.probName)
        # Inform results folder for this run
        self.log.printL('Saving results and log in '+self.log.folderName+'.')
        self.log.prntDashStr()
        # Show parameters for ITman
        self.printPars()
        msg = "\n(You can always change these in the [Settings] " +\
              "section \nof the configuration file)."
        self.log.printL(msg)

        # Default option: generate new solution from scratch
        if self.defOpt == 'newSol':
            msg = "\nDefault starting option (defOpt) is to " + \
                  "generate new initial guess.\n" + \
                  "Hit 'enter' to do it, or any other key to " + \
                  "load a previously started solution."
            inp = self.log.prom(msg)
            if inp == '':

                # TODO: SOL GEN

                # Proceed with solution generation.
                # Find out solution generating mode (default, external, naive)
                # TODO: not all of these options apply to every problem.
                #  Fix this!
                #  A good idea would be to define a list of "standard" modes
                #  in sgra.py and then redefine the modes in each problem
                #  instance (if it is the case).

                self.isNewSol = True
                msg = "\nOk, default mode (initOpt) is '" + self.initOpt + "'."
                msg += "\nHit 'enter' to proceed with it, " + \
                       "or 'd' for 'default',\nor 'n' for 'naive'. " + \
                       "See '" + self.probName + ".py' for details. "
                inp = self.log.prom(msg).lower()
                if inp=='d':
                    self.initOpt='default'
                    self.log.printL("\nProceeding with 'default' mode.\n")
                elif inp=='n':
                    self.initOpt='naive2'
                    self.log.printL("\nProceeding with 'naive2' mode.\n")
                else:
                    self.initOpt='extSol'
                    self.log.printL("\nProceeding with 'extSol' mode.\n" + \
                                    "External file for configurations: " + \
                                    self.confFile + "\n")

                return
            else:

                # TODO: SOL LOAD

                # execution only gets here if the default init is to generate
                # new init guess, but user wants to load solution

                self.isNewSol = False
                self.log.printL("\nGreat, let's load a solution then!")
                # This loop makes sure the user types something that looks like
                # a valid solution.

                # TODO: it should not be so hard to actually check the
                #  existence of the file...
                keepAsk = True
                while keepAsk:
                    msg = "\nThe default path to loading " + \
                          "a solution (loadSolDir) is: " + self.loadSolDir + \
                          "\nHit 'enter' to load it, or type the " + \
                          "path to the alternative solution to be loaded."
                    inp = self.log.prom(msg)
                    if inp == '':
                        keepAsk = False
                    else:
                        # at least an extension check...
                        if inp.lower().endswith('.pkl'):
                            self.loadSolDir = inp
                            keepAsk = False
                        else:
                            self.log.printL('\nSorry, this is not a valid ' + \
                                            "solution path. Let's try again.")
                        #
                    #
                #
                self.log.printL("\nOk, proceeding with " + self.loadSolDir + \
                                "...")
                # Now, ask for a comparing solution.
                self.log.printL("\nOne more question: do you have an " + \
                                "alternative path to load \na solution " + \
                                "that serves as a comparing baseline?")
                keepAsk = True
                while keepAsk:
                    msg = "\nHit 'enter' to use the same path " + "(" + \
                          self.loadSolDir + "),\nor type the path to the" + \
                          " alternative solution to be loaded."
                    inp = self.log.prom(msg)
                    if inp == '':
                        keepAsk = False
                        self.loadAltSolDir = self.loadSolDir
                    else:
                        if inp.lower().endswith('.pkl'):
                            self.loadAltSolDir = inp
                            keepAsk = False
                        else:
                            self.log.printL('\nSorry, this is not a valid ' + \
                                            "solution path. Let's try again.")
                #
                self.log.printL("\nOk, proceeding with\n" +
                                self.loadAltSolDir + "\nas a comparing base!")
                #
            #
        #
        elif self.defOpt == 'loadSol':
            msg = "\nDefault starting option (defOpt) is to load solution." + \
                  "\nThe default path to do it (loadSolDir) is: " + \
                  self.loadSolDir + " .\nHit 'enter' to do it, hit " + \
                  "'I' to generate new initial guess,\n" + \
                  "or type the path to alternative solution to be loaded."
            inp = self.log.prom(msg)
            if inp == '':
                self.isNewSol = False
            elif inp == 'i' or inp == 'I':
                self.isNewSol = True
                self.log.printL("\nOk, generating new initial guess...\n")
                # TODO: there should be a self.initOpt assignment here...
                #  without one, it just defaults to the value in this class's
                #  init method
            else:
                # Solution loading is working!
                self.isNewSol = False
                self.loadSolDir = inp
            return
        else:
            msg = '\nUnknown starting option "' + self.defOpt + \
                            '".\nLeaving now.'
            self.log.printL(msg)
            raise Exception(msg)
        #
    #
    def checkPars(self, sol: sgra.sgra):
        """Makes the user check the parameters of an initial solution,
         and performs an automatic check as well."""

        pLimMin = len(sol.restrictions['pi_min'])
        pLimMax = len(sol.restrictions['pi_max'])

        msg = ''
        if not(pLimMin == pLimMax):
            msg += "\nPi max and min limitation arrays' dimensions mismatch. "
            if pLimMin < pLimMax:
                msg += "\nExtending Pi min limitation array..."
                pi_min = sol.restrictions['pi_min']; val_min = pi_min[-1]
                for i in range(pLimMax-pLimMin):
                    pi_min.append(val_min)
                sol.restrictions['pi_min'] = pi_min
            else:
                msg += "\nExtending Pi max limitation array..."
                pi_max = sol.restrictions['pi_max']; val_max = pi_max[-1]
                for i in range(pLimMin-pLimMax):
                    pi_max.append(val_max)
                sol.restrictions['pi_max'] = pi_max
            #
            pLimMin = len(sol.restrictions['pi_min'])
            pLimMax = len(sol.restrictions['pi_max'])
        #
        # now the limitation arrays are all equal. Check compatibility with p
        if pLimMin < sol.p:
            msg += "\nPi limitation and pi arrays' dimensions mismatch. "
            pi_min = sol.restrictions['pi_min']; val_min = pi_min[-1]
            pi_max = sol.restrictions['pi_max']; val_max = pi_max[-1]
            msg += "\nExtending Pi min and Pi max limitation arrays..."
            for i in range(sol.p-pLimMin):
                pi_min.append(val_min)
                pi_max.append(val_max)
            sol.restrictions['pi_min'] = pi_min
            sol.restrictions['pi_max'] = pi_max

        if len(msg) > 0:
            msg += "\nPress any key to continue...\n"
            self.log.prom(msg)

        # Ok, now check if the given pi's respect the limitations. If some of
        # them does not, then it's pretty much game over...

        msg = ''
        Fail = False
        for i in range(sol.p):
            thisPiMin = sol.restrictions['pi_min'][i]
            if thisPiMin is not None:
                if sol.pi[i] < thisPiMin:
                    msg += "\nPi[" + str(i) + "] < pi_min."
                    Fail = True
            thisPiMax = sol.restrictions['pi_max'][i]
            if thisPiMax is not None:
                if sol.pi[i] > thisPiMax:
                    msg += "\nPi[" + str(i) + "] > pi_max."
                    Fail = True
        if Fail:
            msg += '\nExiting the program.'
            self.log.printL(msg)
            raise Exception(msg)

        # All tests ok! Let's proceed to the usual parameter checking.
        keepLoop = True
        while keepLoop:
            self.log.prntDashStr()
            sol.printPars()
            #sol.plotSol()
            self.log.prntDashStr()
            self.bell(lag=1e-8,nRing=1)
            msg = "\nAre these parameters OK?\n" + \
                  "Press 'enter' to continue, or update the configuration " + \
                  "file:\n    (" + self.confFile + ")\n" + \
                  "and then press any other key to reload the " + \
                  "parameters from that file.\n" + \
                  "Please notice that this only reloads parameters, not " + \
                  "necessarily\nregenerating the initial guess.\n" + \
                  "(See 'loadParsFromFile' method in 'sgra.py'.)"
            inp = self.log.prom(msg)
            if inp != '' or not self.isManu:
                self.log.printL("\nFine, reloading parameters...")
                sol.loadParsFromFile(file=self.confFile)
                if not self.isManu:
                    keepLoop = False
            else:
                self.log.printL("\nGreat! Moving on then...\n")
                keepLoop = False
            #
        #

    def loadSol(self, path=''):
        """Loads a solution from a .pkl file."""
        if path == '':
            path = self.loadSolDir

        self.log.printL("\nReading solution from '"+path+"'.")
        with open(path, 'rb') as inpt:
            sol = dill.load(inpt)

        # By default the logger is not saved with the solution
        sol.log = self.log

        # For all intents and purposes, nothing is updated anymore
        sol.isUpdated.setAll(False)

        return sol

    def saveSol(self, sol: sgra.sgra, path=''):
        """Saves solution to a .pkl file."""
        if path == '':
            path = self.probName + '_sol_' + getNowStr() + '.pkl'

        # set the logger to a manual one for loading later
        sol.log = logger(sol.probName, makeDir=False, mode='screen')
        self.log.printL("\nWriting solution to '"+path+"'.")
        with open(path, 'wb') as outp:
            dill.dump(sol,outp,-1)

        sol.log = self.log

    def setInitSol(self, sol: sgra.sgra):
        """Setup for the initial solution."""
        msg = "Setting initial solution.\n" + \
              "Please wait, you will be asked to confirm it later.\n\n"
        self.log.printL(msg)
        if self.isNewSol:
            # declare problem:
            solInit = sol.initGues({'initMode':self.initOpt,
                                    'confFile':self.confFile})
            self.log.printL("Saving a copy of the configuration file " + \
                            "in this run's folder.")
            self.confFile = shutil.copy2(self.confFile, self.log.folderName +
                                         os.sep)
            self.saveSol(sol, self.log.folderName + os.sep + 'solInit.pkl')
        else:
            # load previously prepared solution
            self.log.printL('Loading "current" solution...')
            sol = self.loadSol()
            self.log.printL("Saving a copy of the default configuration " + \
                            "file in this run's folder.\n(Just for " + \
                            "altering later, if necessary).")
            self.confFile = shutil.copy2(self.confFile, self.log.folderName + \
                                         os.sep)
            self.log.printL('Loading "initial" solution ' + \
                            '(for comparing purposes only)...')
            solInit = self.loadSol(path=self.loadAltSolDir)

        # Prepare error-calculating matrix (used in each call of calcErr and calcQ)
        sol.prepCalcErrMat()
        # Plot obtained solution, check parameters
        self.log.prntDashStr()
        self.log.printL("\nProposed initial guess:\n")
        sol.plotSol()
        self.checkPars(sol)

        # Calculate P values, store them
        P,Pint,Ppsi = sol.calcP()
        sol.histP[sol.NIterRest] = P
        sol.histPint[sol.NIterRest] = Pint
        sol.histPpsi[sol.NIterRest] = Ppsi

        if sol.NIterGrad == 0:
            # Properly record the initial value of I (it does make sense!)
            sol.I,Iorig,Ipf = sol.calcI()
            sol.histI[0] = sol.I
            sol.histIorig[0] = Iorig
            sol.histIpf[0] = Ipf

        # Plot trajectory
        sol.plotTraj()

        # TODO: these parameters should go to an external file!
        # Setting debugging options (rest and grad).
        # Declaration is in sgra.py!
        sol.dbugOptRest.setAll(opt=self.default_dbugOptRest)
        sol.dbugOptGrad.setAll(opt=self.default_dbugOptGrad)
#        sol.log = self.log
#        solInit.log = self.log
        return sol,solInit

    def showHistPCond(self, sol: sgra.sgra):
        if sol.NIterRest % self.RestHistShowRate == 0:
            return True
        else:
            return False

    def plotSolRestCond(self, sol: sgra.sgra):
        if sol.NIterRest % self.RestPlotSolRate == 0:
            return True
        else:
            return False

    def restRnds(self, sol: sgra.sgra):
        self.log.printL("\nStart of restoration rounds. " + \
                        "P = {:.4E}".format(sol.P))
        contRest = 0
        origDbugOptRest = sol.dbugOptRest.copy()

        while sol.P > sol.tol['P']:
            sol.rest(parallelOpt=self.parallelOpt)
            contRest += 1
            #sol.plotSol()
            #input("\nOne more restoration complete.")
        #

#        sol.showHistP()
#        self.log.printL("\nEnd of restoration rounds (" + str(contRest) + \
#                        "), P = {:.4E}".format(sol.P) + ". Solution so far:")
#        sol.plotSol()
        self.log.printL("\nEnd of restoration rounds (" + str(contRest) + \
                        "), P = {:.4E}".format(sol.P))
        sol.dbugOptRest.setAll(opt=origDbugOptRest)
        return sol, contRest

    def frstRestRnds(self, sol: sgra.sgra):
        self.log.prntDashStr()
        self.log.printL("\nBeginning first restoration rounds...\n")
        sol.P,_,_ = sol.calcP(mustPlotPint=True)
        sol, contRest = self.restRnds(sol)

        self.saveSol(sol,self.log.folderName + os.sep + 'solInitRest.pkl')

        return sol

    def showHistQCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.GradHistShowRate == 0:
            return True
        else:
            return False

    def showHistICond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.GradHistShowRate == 0:
            return True
        else:
            return False

    def showHistGradStepCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.GradHistShowRate == 0:
            return True
        else:
            return False

    def showHistGRrateCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.ShowGRrateRate == 0:
            return True
        else:
            return False

    def showEigCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.ShowEigRate == 0:
            return True
        else:
            return False

    def showLambCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.ShowLambRate == 0:
            return True
        else:
            return False

    def showVarCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.ShowVarRate == 0:
            return True
        else:
            return False

    def plotSolGradCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.GRplotSolRate == 0:
            return True
        else:
            return False

    def saveSolCond(self, sol: sgra.sgra):
        #return False

        if sol.NIterGrad % self.GRsaveSolRate==0:
            return True
        else:
            return False

    def gradRestPausCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.GRpausRate==0:
            return True
        else:
            return False

    def plotResPCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.plotResPRate == 0:
            return True
        else:
            return False

    def plotResQCond(self, sol: sgra.sgra):
        if sol.NIterGrad % self.plotResQRate == 0:
            return True
        else:
            return False

    def gradRestCycl(self, sol: sgra.sgra, altSol:sgra.sgra = None):
        """Gradient-restoration cycle."""

        self.log.prntDashStr()
        self.log.printL("\nBeginning gradient-restoration rounds...")
        evnt = 'init'
        do_GR_cycle = True
        last_grad = 0
        next_grad = 0
        low_step = False
        plotResP, plotResQ = False, False
        stepMan = None
        while do_GR_cycle:

            # Start of new cycle: calc P, I, as well as a new grad correction
            # in order to update lambda and mu, and therefore calculate Q for
            # checking.

            sol.P,_,_ = sol.calcP(mustPlotPint=plotResP)

            P_base = sol.P
            sol.J, sol.J_Lint, sol.J_Lpsi, sol.I, sol.Iorig, sol.Ipf = sol.calcJ()
            I_base = sol.I
            self.log.prntDashStr()
            msg = "\nStarting new cycle, I_base = {:.4E}".format(I_base) + \
                  ", P_base = {:.4E}".format(P_base)
            self.log.printL(msg)

            isParallel = self.parallelOpt.get('gradLMPBVP',False)
            corr, lam, mu = sol.LMPBVP(rho=1.0,isParallel=isParallel)
            sol.lam, sol.mu = lam, mu

            sol.Q, sol.Qx, sol.Qu, sol.Qp, sol.Qt = sol.calcQ(mustPlotQs=plotResQ)
            corr['dJdStepTheo'] = -sol.Q

            if sol.Q <= sol.tol['Q'] or sol.NIterGrad >= self.MaxIterGrad or low_step:
                # Run again calcQ, making sure the residuals are plotted.
                # This is good for debugging eventual convergence errors
                #_,_,_,_,_ = sol.calcQ(mustPlotQs=True)
                if sol.Q <= sol.tol['Q']:
                    msg = '\nTolerance for Q functional was met!'
                elif low_step:
                    msg = '\nGradient step search returned a value too low.'
                else:
                    msg = '\nToo many gradient iterations.'
                msg += "\nTerminate program. Solution is sol_r."
                self.log.printL(msg)
                # This is just to make sure the final values of Q, I, J, etc
                # get registered.
                sol.updtEvntList(evnt)
                sol.updtHistGrad(0.,1) # why is this here?
                sol.updtHistP(mustPlotPint=True)
                do_GR_cycle = False
            else:
                msg = "\nOk, now Q = {:.4E} > {:.4E} = tolQ,\nso let's keep" \
                      " improving the solution!\n".format(sol.Q,sol.tol['Q'])
                self.log.printL(msg)
                #sol.plotSol()
                #sol.plotF()
                #sol.plotTraj()
                #input("\nVamos tentar dar um passo de grad pra frente!")

                keep_walking_grad = True
                retry_grad = False
                #alfa_g_0 = 1.0
                alfa_base = sol.histStepGrad[sol.NIterGrad]

                while keep_walking_grad:
                    # This stays here in order to properly register the
                    # gradAccepts and the gradRejects
                    sol.updtEvntList(evnt)

                    # DEBUG: start plotting
                    #if sol.NIterGrad > 11:
                    #     sol.dbugOptGrad['plotCalcStepGrad'] = True

                    alfa, sol_new, stepMan = sol.grad(corr, alfa_base,
                                                      retry_grad, stepMan)
                    # BEGIN_DEBUG:
                    I_mid,_,_ = sol_new.calcI()
                    P_mid,_,_ = sol_new.calcP()
                    # END_DEBUG
                    sol_new, contRest = self.restRnds(sol_new)
                    # Up to this point, the solution is fully restored!

                    sol_new.I,_,_ = sol_new.calcI()
                    msg = "\nBefore:\n" \
                          "  I = {:.6E}, P = {:.4E}\n... ".format(I_base,P_base) + \
                          "after grad (with {} obj evals)".format(stepMan.cont+1) + \
                          " gave alfa = {:.4E}:\n".format(alfa) + \
                          "  dI = {:.6E}, P = {:.4E}".format(I_mid-I_base,P_mid) + \
                          "\n... and after restoring {} times:".format(contRest) + \
                          "\n  dI = {:.6E}".format(sol_new.I-I_base) + \
                          ", P = {:.4E}".format(sol_new.P) + \
                          "\nVariation in I: " + \
                            str(100.0*(sol_new.I/I_base-1.0)) + "%" + \
                          '\nRelative reduction of I' \
                          ' (w.r.t. theoretical dI/dStep): ' + \
                          str(100.0 * ((sol_new.I-I_base)/alfa/(-sol.Q))) + "%"
                    self.log.printL(msg)

                    if alfa < 1e-30:
                        # The chosen step value is too low. There will be problems...
                        # It is better to leave right now.
                        sol = sol_new
                        evnt = 'gradOK'
                        keep_walking_grad = False
                        low_step = True
                        next_grad += 1
                        msg = "\nNext grad counter = {}" \
                              "\nLast grad counter = {}" \
                              "\nStep is too low. Leaving...".format(next_grad,
                                                                     last_grad)
                        self.log.printL(msg)
                    elif sol_new.I <= I_base:
                        # Here, the I functional has been lowered!
                        # The grad step is accepted:
                        # rebase the solution and leave this loop.

                        sol = sol_new
                        evnt = 'gradOK'
                        keep_walking_grad = False
                        next_grad += 1
                        msg = "\nNext grad counter = {}" \
                              "\nLast grad counter = {}" \
                              "\nI was lowered, step given!".format(next_grad,
                                                                    last_grad)
                        self.log.printL(msg)
                    else:
                        # The conditions were not met. Discard this solution
                        # and try a new gradStep on the previous baseline

                        last_grad += 1
                        retry_grad = True
                        evnt = 'gradReject'
                        # Save in 'sol' the histories from sol_new,
                        # otherwise the last grad and the rests would be lost!
                        sol.copyHistFrom(sol_new)
                        alfa_base = alfa
                        msg = "\nNext grad counter = {}" \
                              "\nLast grad counter = {}\nI was not lowered... " \
                              "trying again!".format(next_grad,last_grad)
                        self.log.printL(msg)
                    sol.histObjEval[next_grad + last_grad] = stepMan.cont+1
                    #
                    #input("Press any key to continue... ")
                #
                if retry_grad:
                    sol.rest(parallelOpt=self.parallelOpt)
                #
            #

            # Check for histories/solution showing conditions, etc.
            if self.showHistQCond(sol):
                self.log.printL("\nHistQ showing condition is met!")
                sol.showHistQ()
                sol.showHistQvsI()

            if self.showHistICond(sol):
                self.log.printL("\nHistI showing condition is met!")
                sol.showHistI()

            if self.showHistGradStepCond(sol):
                self.log.printL("\nHistGradStep showing condition is met!")
                sol.showHistGradStep()

            if self.showHistGRrateCond(sol):
                self.log.printL("\nHistGRrate showing condition is met!")
                sol.showHistGRrate()

            if self.showEigCond(sol):
                self.log.printL("\nEigenvalue showing condition is met!\n" + \
                                "It will be done in the next gradStep.")
                sol.save['eig'] = True
            else:
                sol.save['eig'] = False

            if self.showLambCond(sol):
                self.log.printL("\nLambda showing condition is met!\n" + \
                                "It will be done in the next gradStep.")
                sol.save['lambda'] = True
            else:
                sol.save['lambda'] = False

            if self.showVarCond(sol):
                self.log.printL("\nVariation showing condition is met!\n" + \
                                "It will be done in the next gradStep.")
                sol.save['var'] = True
            else:
                sol.save['var'] = False

            if self.saveSolCond(sol):
                self.log.printL("\nSolution saving condition is met!")
                #self.log.prntDashStr()
                name = self.log.folderName + os.sep + \
                       'sol-{}gradIts.pkl'.format(sol.NIterGrad)
                self.saveSol(sol,name)

            if self.plotSolGradCond(sol):
                #self.log.prntDashStr()
                self.log.printL("\nSolution showing condition is met!")
                self.log.printL("\nSolution so far:")
                sol.plotSol()
                sol.plotF()

                if altSol is not None:
                    sol.compWith(altSol,'Initial guess')
                    sol.compWith(altSol,'Initial guess',piIsTime=False)
#                    sol.plotTraj(True,altSol,'Initial guess',\
#                                 mustSaveFig=False)
                    sol.plotTraj(True,altSol,'Initial guess')
                else:
                    sol.plotTraj()

            if self.gradRestPausCond(sol):
                print("\a")
                self.log.prntDashStr()
                self.log.printL(str(datetime.datetime.now()))
                msg = "\nAfter " + str(sol.NIterGrad) + \
                      " gradient iterations,\n" + \
                      "Grad-Rest cycle pause condition has been reached.\n" + \
                      "Press any key to continue, or ctrl+C to stop.\n" + \
                      "Load last saved solution to go back to GR cycle."
                self.log.prom(msg)

            if self.plotResPCond(sol):
                self.log.printL("\nP residual showing condition is met!\n" + \
                                "It will be done in the next gradStep.")
                plotResP = True
            else:
                plotResP = False

            if self.plotResQCond(sol):
                self.log.printL("\nQ residual showing condition is met!\n" + \
                                "It will be done in the next gradStep.")
                plotResQ = True
            else:
                plotResQ = False
            #
        #

        return sol
