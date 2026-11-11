import xml.etree.ElementTree as ET
from .utilis import iterNodes
import os
import re
import warnings
class Handler:
    """"
    Handler class for processing metadata from a zip file containing product information.

    Attributes:
        filepath (str): Path to the zip file containing the product data.
        PRF (float): Pulse repetition frequency extracted from the metadata.
        AzimBand (float): Total azimuth bandwidth extracted from the metadata.
        RangeBand (float): Total range bandwidth extracted from the metadata.
        ChirpBand (float): Chirp bandwidth calculated from pulse length and ramp rate.
        AzimRes (int): Azimuth resolution (default is 5).
        RangeRes (int): Range resolution (default is 20).
        RangeSpacing (float): Range pixel spacing extracted from the metadata.
        AzimSpacing (float): Azimuth pixel spacing extracted from the metadata.
        WeightFunctRangeParams (float): Range window coefficient for weighting function.
        WeightFunctRange (str): Type of weighting function used for range processing (default is 'HAMMING').
        WeightFunctAzimParams (float): Azimuth window coefficient for weighting function.
        WeightFunctAzim (str): Type of weighting function used for azimuth processing (default is 'HAMMING').
        CentralFreqRange (int): Central frequency for range processing (default is 0).
        CentralFreqAzim (int): Central frequency for azimuth processing (default is 0).

    Methods:
        __init__(filepath):
            Initializes the Handler object with the path to the zip file.

        Chain():
            Processes the metadata from the zip file, extracting relevant parameters
            such as PRF, bandwidths, resolutions, and spacing. The method reads the
            XML annotation file within the zip archive and parses the required data.
    """
    def __init__(self, metadata_pointer, swath=None, polarization=None, require_unique=None):
        self.swath = swath.upper() if swath else None
        self.polarization = polarization.upper() if polarization else None
        # Keep selector-based strictness by default for direct Handler users,
        # while allowing higher-level legacy workflows to explicitly opt out
        # when the band name does not identify a swath.
        self.require_unique = (
            bool(self.swath or self.polarization)
            if require_unique is None
            else bool(require_unique)
        )
        self.annotation_file = None
        
        if type(metadata_pointer) is str:
            self.filepath = metadata_pointer
            self.assetDict = None
        elif type(metadata_pointer) is dict:
            self.assetDict = metadata_pointer
            self.filepath = None
        else:
            raise ValueError("Unsupported metadata format, please pass file path or nested dictionary.")

  
    def chain(self):
        """
        Main method to process metadata from the .SAFE directory.
        """
        if self.filepath: 
            try:
                annotations = self._get_annotations()
                xml = self._read_annotation(annotations)
                root = ET.fromstring(xml)
                self._extract_metadata_from_xml(root)
            except Exception as e:
                raise RuntimeError(f"Error processing metadata: {e}")
        else:
            self._extract_metadata_from_dict()

    def _open_directory(self):
        """
        Verifies that the .SAFE directory exists and is accessible.
        """
        try:
            if not os.path.isdir(self.filepath):
                raise FileNotFoundError(f"Directory not found: {self.filepath}")
            return self.filepath
        except Exception as e:
            raise RuntimeError(f"Failed to access .SAFE directory: {e}")


    def _get_annotations(self):
        """
        Retrieves the list of annotation files from the .SAFE directory.
        """
        try:
            # First verify the directory exists
            self._open_directory()
            
            # Find all XML files in annotation folders
            annotations = []
            for root, dirs, files in os.walk(self.filepath):
                if "annotation" in root and "calibration" not in root and "rfi" not in root:
                    for file in files:
                        if file.endswith(".xml"):
                            annotations.append(os.path.join(root, file))
            
            if not annotations:
                raise ValueError("No valid annotation files found in the .SAFE directory.")
            
            return annotations
        except Exception as e:
            raise RuntimeError(f"Failed to get annotations: {e}")


    def _read_annotation(self, annotations):
        """
        Read the unique annotation matching the requested swath and polarization.
        """
        matches = []
        for path in annotations:
            tokens = re.split(r"[-_.]", os.path.basename(path).upper())
            if self.swath and self.swath not in tokens:
                continue
            if self.polarization and self.polarization not in tokens:
                continue
            matches.append(path)
        if not matches or (self.require_unique and len(matches) != 1):
            raise ValueError(
                f"Expected one annotation for swath={self.swath!r}, "
                f"polarization={self.polarization!r}; found {len(matches)}: {matches}"
            )
        if len(matches) > 1:
            message = (
                "Multiple SAFE annotations match the supplied selectors; "
                "selecting the first for backward compatibility. "
                "Supply more specific selectors to avoid ambiguity."
            )
            warnings.warn(message, UserWarning, stacklevel=3)
        self.annotation_file = matches[0]
        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as file:
                return file.read()
        except OSError as e:
            raise RuntimeError(f"Failed to read annotation file: {e}")


    def _extract_metadata_from_xml(self, root):
        """
        Extracts metadata from the XML root and assigns it to class attributes.
        """
        try:
            ValDict = {}
            Ancillary = iterNodes(root, Val_Dict=ValDict)

            self.PRF = float(Ancillary.get('prf', 0))
            self.AzimBand = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//azimuthProcessing//totalBandwidth').text or 0)
            self.RangeBand = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//rangeProcessing//totalBandwidth').text or 0)
            self.ChirpBand = float(Ancillary.get('txPulseLength', 0)) * float(Ancillary.get('txPulseRampRate', 0))
            self.RangeSpacing = float(Ancillary.get('rangePixelSpacing', 0))
            self.AzimSpacing = float(Ancillary.get('azimuthPixelSpacing', 0))
            self.WeightFunctRangeParams = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//rangeProcessing//windowCoefficient').text or 0)
            self.WeightFunctAzimParams = float(root.find('.//imageAnnotation//processingInformation//swathProcParams//azimuthProcessing//windowCoefficient').text or 0)

        except AttributeError as e:
            raise ValueError(f"Missing or invalid XML elements: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting metadata: {e}")
    
    def _extract_metadata_from_dict(self):
        """
        Extracts metadata from nested dictionary of asset metadata. Added for use in AssetToZarr pipeline.
        
        """
        
        self.PRF = float(self.assetDict["generalAnnotation"]["downlinkInformationList"]["downlinkInformation"]["prf"])
        self.AzimBand = float(self.assetDict["imageAnnotation"]["processingInformation"]["swathProcParamsList"]["swathProcParams"]["azimuthProcessing"]["totalBandwidth"])
        self.RangeBand = float(self.assetDict["imageAnnotation"]["processingInformation"]["swathProcParamsList"]["swathProcParams"]["rangeProcessing"]["totalBandwidth"])
        self.ChirpBand = float(self.assetDict["generalAnnotation"]["downlinkInformationList"]["downlinkInformation"]["downlinkValues"]["txPulseLength"]) * float(self.assetDict["generalAnnotation"]["downlinkInformationList"]["downlinkInformation"]["downlinkValues"]["txPulseRampRate"])
        self.RangeSpacing = float(self.assetDict["imageAnnotation"]["imageInformation"]["rangePixelSpacing"])
        self.AzimSpacing = float(self.assetDict["imageAnnotation"]["imageInformation"]["azimuthPixelSpacing"])
        self.WeightFunctRangeParams = float(self.assetDict["imageAnnotation"]["processingInformation"]["swathProcParamsList"]["swathProcParams"]["rangeProcessing"]["windowCoefficient"])
        self.WeightFunctAzimParams = float(self.assetDict["imageAnnotation"]["processingInformation"]["swathProcParamsList"]["swathProcParams"]["azimuthProcessing"]["windowCoefficient"])

        
