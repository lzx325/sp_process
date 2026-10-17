//import qupath.lib.scripting.QP
import java.nio.file.Paths
import qupath.lib.common.ThreadTools
ThreadTools.setParallelism(96)


// Project Dir
def project = getProject()
def project_dir = project.getPath().parent
// qupath_scripts/ dir, resolved from this script's path (qupath_scripts/scripts/<this>.groovy).
// QuPath sets "qupath.script.file" when the script is run from a saved file (script editor or CLI).
if (!binding.hasVariable("qupath.script.file"))
    throw new RuntimeException("Cannot resolve script location: save the script to a file and run it from there")
def code_dir = Paths.get(binding.getVariable("qupath.script.file")).toAbsolutePath().parent.parent

// Get the current image data
def imageData = getCurrentImageData()

// Get the ImageServer
def imageServer = imageData.getServer()

// Get the PixelType
def pixelType = imageServer.getPixelType()

def imageType = "HE"
def json_fp = ""
// Load thresholder json
switch(pixelType) {
    case PixelType.UINT8:
        json_fp = code_dir.resolve(String.format("classifiers/pixel_classifiers/tissue_detection_%s_uint8.json",imageType)).toString()
        break
    case PixelType.UINT16:
        json_fp = code_dir.resolve(String.format("classifiers/pixel_classifiers/tissue_detection_%s_uint16.json",imageType)).toString()
        break
    default:
        throw new RuntimeException("Unknown dtype")
}

def json = new File(json_fp).text

// clear all objects
removeAllObjects()

double minArea = 1000
double minHoleArea = 1000
//
//double minArea = 10r
//double minHoleArea = 10

//double minArea = 2000
//double minHoleArea = 100

// Use thresholder to detect tissue region


def thresholder = GsonTools.getInstance().fromJson(json, qupath.lib.classifiers.pixel.PixelClassifier.class)

createAnnotationsFromPixelClassifier(thresholder, minArea, minHoleArea)

selectObjectsByClassification("Tissue");