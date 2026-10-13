import java.nio.file.Paths

def gson = GsonTools.getInstance(true)
def annotations = getAnnotationObjects()

def project = getProject()
def project_dir = project.getPath().parent

def out_dir = project_dir.resolve("external_data/tissue_detection/geojson")

def entry = getProjectEntry()
def uri = entry.readImageData().getServer().getURIs()[0]
def image_name = Paths.get(uri).getFileName().toString().split("\\.")[0]

def dir = new File(out_dir.toString())
if (!dir.exists()) {
    dir.mkdirs()
}

def out_fp = out_dir.resolve(String.format("%s.geojson",image_name)).toString()
File file = new File(out_fp)
 file.withWriter('UTF-8') {
     gson.toJson(annotations,it)
 }