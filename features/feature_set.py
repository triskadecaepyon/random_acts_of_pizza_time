import os
import abc
import logging
import sys
import inspect
import hashlib
import json


class FeatureSet:
    __metaclass__ = abc.ABCMeta

    def __init__(self, useBinaryFileMode=False, verbose=False):
        self.fileModeModifier = 'b' if useBinaryFileMode else ''

        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def smartLoad(self, outputFilePath = None, **kwargs):
        if not outputFilePath:
            outputFileDir = os.path.dirname(inspect.getsourcefile(self.__class__))
            outputFilePath = os.path.join(outputFileDir, self.__class__.__name__) + ".fts" # fts -> _f_ea_t_ure _s_et

        # ensures that we have unique files based on the kwargs
        if kwargs:
            md5Hasher = hashlib.md5()
            kwargsAsJSON = json.dumps(kwargs, sort_keys = True)
            md5Hasher.update(kwargsAsJSON)
            kwargsAsHash = md5Hasher.hexdigest().upper()[:16]

            (root, ext) = os.path.splitext(outputFilePath)
            outputFilePath = '%s-%s%s' % (root, kwargsAsHash, ext)
            outputFileDescriptionPath = '%s-%s-kwargs' % (root, kwargsAsHash) + '.txt'
            
        outputFileExists = os.path.exists(outputFilePath)
        self.logger.debug("checking if output file exists: %s" % outputFileExists)

        if outputFileExists:
            outputFileIsOlder = self.__isOutputFileOlder(outputFilePath)
            self.logger.debug("checking if output file is older: %s" % outputFileIsOlder)

        if not outputFileExists or outputFileIsOlder:
            outputFileDir = os.path.dirname(outputFilePath)
            if outputFileDir:
                try: 
                    os.makedirs(outputFileDir)
                except OSError:
                    if not os.path.isdir(outputFileDir):
                        raise

            self.logger.debug("extracting features -> %s" % outputFilePath)
            with open(outputFilePath, 'w' + self.fileModeModifier) as outputFile:
                # *** optimization ***
                # if _extract() returns the features in addition to writing them to a file,
                # we will just return them here so we don't have to call _load() below.
                # However, this is not a requirement of the _extract() method
                features = self._extract(outputFile, **kwargs)
            
            if kwargs:
                self.logger.debug("creating description file-> %s" % outputFileDescriptionPath)
                with open(outputFileDescriptionPath, 'w') as outputFile:
                    outputFile.write(kwargsAsJSON)
                    outputFile.write(os.linesep)

            if features is not None:
                self.logger.debug("loading features <- %s" % outputFilePath)
                return features

        else:
            self.logger.debug("skipping feature extraction")

        self.logger.debug("loading features <- %s" % outputFilePath)
        with open(outputFilePath, 'r' + self.fileModeModifier) as outputFile:
            return self._load(outputFile)

    def __isOutputFileOlder(self, outputFilePath):

        # This is a bit hacky, but it works!
        classFilePaths = [os.path.abspath(inspect.getsourcefile(base)) for base in self.__class__.__bases__]
        classFilePaths.append(os.path.abspath(inspect.getsourcefile(self.__class__)))

        outputFileModificationTime = os.path.getmtime(outputFilePath) 

        # Extract features if the output file is older than any of the files containing classes derived from this
        # class. In other words, if any file containing a derived FeatureSet class is changed, the output file will
        # be older and we will need to reextract features
        for classFilePath in classFilePaths:
            classFileModificationTime = os.path.getmtime(classFilePath)
            if classFileModificationTime > outputFileModificationTime:
                return True

        return False

    @abc.abstractmethod
    def _extract(self, outputFile, **kwargs):
        """ 
        Subclasses must implement this method.
        It is intended to extract features to an output file.
        Note: as an optimization, the _extract() method can immediately return the features it has
            extracted after writing them to the output file
        """
        return

    @abc.abstractmethod
    def _load(self, outputFile):
        """
        Subclasses must implement this method.
        It is intended to load and return features from an output file.
        """
        return
