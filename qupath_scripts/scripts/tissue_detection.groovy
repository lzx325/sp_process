//import qupath.lib.scripting.QP
import java.nio.file.Paths
import qupath.lib.common.ThreadTools
ThreadTools.setParallelism(96)


// Project Dir
def project = getProject()
def project_dir = project.getPath().parent
def code_dir = Paths.get("/mnt/radonc-Li02_vol2/public/he_codex/raw/qupath_scripts/")

// Get the current image data
def imageData = getCurrentImageData()

// Get the ImageServer
def imageServer = imageData.getServer()

// Get the PixelType
def pixelType = imageServer.getPixelType()

def imageType = "mIHC"
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

double minArea = 20000
double minHoleArea = 10000
//
//double minArea = 10r
//double minHoleArea = 10

//double minArea = 2000
//double minHoleArea = 100

// Use thresholder to detect tissue region


def thresholder = GsonTools.getInstance().fromJson(json, qupath.lib.classifiers.pixel.PixelClassifier.class)

createAnnotationsFromPixelClassifier(thresholder, minArea, minHoleArea)

selectObjectsByClassification("Tissue");