import numpy as np
import json
class QuPathGeojson:
    def __init__(self):
        self.cell_polygon = None
        self.nucleus_polygon = None
        self.classification = None
        self.measurements = None
        self.cell_id = None

    @staticmethod
    def generate_cell_id(n_cells):
        import uuid
        cell_id = [str(uuid.uuid4()) for _ in range(n_cells)]
        return cell_id
    
    def write_geojson(self,json_fp):
        assert self.cell_polygon is not None

        n_cells = len(self.cell_polygon)
        cell_indices = set(self.cell_polygon.keys()) 

        assert self.classification is None or len(self.classification)==n_cells and set(self.classification.keys()) == cell_indices
        assert self.measurements is None or len(self.measurements)==n_cells and set(self.measurements.keys()) == cell_indices
        assert self.cell_id is None or len(self.cell_id)==n_cells and set(self.cell_id.keys()) == cell_indices

        if self.cell_id is None:
            self.cell_id = dict(zip (cell_indices, self.generate_cell_id(n_cells)))
        
        json_data = list()
        for i, cell_id in self.cell_id.items():
            rec = {
                "type": "Feature",
                "id": cell_id
            }
            if self.cell_polygon is None and self.nucleus_polygon is not None:
                self.cell_polygon = self.nucleus_polygon
                
            if self.cell_polygon is not None:
                rec["geometry"]={
                    "type":"Polygon",
                    "coordinates": [self.cell_polygon[i].tolist()]
                }

            if self.nucleus_polygon is not None and i in self.nucleus_polygon:
                rec["nucleusGeometry"]={
                    "type":"Polygon",
                    "coordinates": [self.nucleus_polygon[i].tolist()]
                }

            properties = {"objectType":"cell"}
            if self.classification is not None:
                properties["classification"]=self.classification[i]
            if self.measurements is not None:
                properties["measurements"]=self.measurements[i]
            rec["properties"] = properties
            json_data.append(rec)
            
        with open(json_fp, "w") as f:
            json.dump(json_data, f, indent=2)
    
    def read_geojson(self,json_fp):
        import warnings
        with open(json_fp, "r") as f:
            json_data = json.load(f)
        
        self.cell_polygon = dict()
        self.nucleus_polygon = dict()
        self.classification = dict()
        self.measurements = dict()
        self.cell_id = dict()
        
        for i, rec in enumerate(json_data):
            if rec["geometry"]["type"] != "Polygon":
                warnings.warn("Geometry type is not Polygon at entry {}: {}".format(i, rec["geometry"]["type"]), UserWarning)
                continue
            else:
                if "geometry" in rec and rec["geometry"]["type"] == "Polygon":
                    self.cell_polygon[i] = np.array(rec["geometry"]["coordinates"][0])

                if "nucleusGeometry" in rec and rec["nucleusGeometry"]["type"] == "Polygon":
                    self.nucleus_polygon[i] = np.array(rec["nucleusGeometry"]["coordinates"][0])

                if "classification" in rec["properties"]:
                    self.classification[i] = rec["properties"]["classification"]

                if "measurements" in rec["properties"]:
                    self.measurements[i] = (rec["properties"]["measurements"])

                self.cell_id[i] = rec["id"]

                if len(self.cell_polygon) != len(self.cell_id):
                    raise ValueError("Number of cell polygons does not match number of cell IDs. At entry {} has {} polygons and {} IDs.".format(i, len(self.cell_polygon), len(self.cell_id)))
    
    def get_integer_mask(self, image_height, image_width):
        import cv2 as cv
        mask = np.zeros((image_height, image_width), dtype=np.int32)
        for i, polygon in self.cell_polygon.items():
            cv.fillPoly(mask, [polygon.astype('int64')], color=i+1)
        return mask

def save_stardist_prediction(res_dict,save_fp):
    qpjson = QuPathGeojson()
    nucleus_coords = res_dict["coord"].transpose([0,2,1])[:,:,[1,0]]
    nucleus_coords = np.concatenate([nucleus_coords,nucleus_coords[:,0:1,:]],axis = 1)
    qpjson.nucleus_polygon = nucleus_coords
    qpjson.classification = [{"name":"Segmented*", "color": [200,200,200] } for i in range(len(nucleus_coords))]
    qpjson.measurements = [{"Segmentation Prob": res_dict["prob"][i].item() } for i in range(len(res_dict["prob"]))]
    qpjson.write_geojson(save_fp)

def cell_centroid_dataframe_to_geojson(df, output_file, x_col='x', y_col='y', classification_col=None, color_col=None):
    """
    Convert a DataFrame with x,y coordinates to a GeoJSON file.

    Parameters:
        df (pd.DataFrame): Input DataFrame with centroid coordinates.
        x_col (str): Name of the column containing x coordinates.
        y_col (str): Name of the column containing y coordinates.
        classification_col (str, optional): Name of the column containing classification strings (e.g., 'cell_type').
        color_col (str, optional): Name of the column containing hex color strings (e.g., '#FF0000').
        output_file (str): Path to save the GeoJSON file.
    """
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point

    # Create geometry from x,y
    gdf = pd.DataFrame(index=df.index)
    
    gdf['geometry'] = [Point(xy).buffer(2.5) for xy in zip(df[x_col], df[y_col])]
    
    # Create nested properties
    classification_properties = []
    for idx in df.index:
        classification_prop = {}
        
        # Add classification as nested property if specified
        if classification_col is not None:
            if classification_col not in df.columns:
                raise ValueError(f"Column '{classification_col}' not found in DataFrame")
            classification_name = df.loc[idx, classification_col]
            classification_prop['name'] = classification_name
            
            # Add color to classification if color_col is specified
        if color_col is not None:
            if color_col not in df.columns:
                raise ValueError(f"Column '{color_col}' not found in DataFrame")
            color_value = df.loc[idx, color_col]
            classification_prop['color'] = color_value
        
            
        classification_properties.append(classification_prop)
    
    # Add properties to GeoDataFrame
    gdf['classification'] = classification_properties
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    # Export to GeoJSON
    gdf.to_file(output_file, driver='GeoJSON')

def polygon_to_geojson(shapes_list, metadata_df, output_file, classification_col=None, color_col=None):
    """
    Convert a list of polygon geometries with metadata DataFrame to a GeoJSON file.

    Parameters:
        shapes_list (list): List of shapely.geometry.Polygon objects.
        metadata_df (pd.DataFrame): DataFrame containing metadata for the polygons.
        classification_col (str, optional): Name of the column containing classification strings (e.g., 'cell_type').
        color_col (str, optional): Name of the column containing hex color strings (e.g., '#FF0000').
        output_file (str): Path to save the GeoJSON file.
    """
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    import warnings

    # Validate inputs
    if not isinstance(shapes_list, list):
        raise ValueError("shapes_list must be a list")
    
    if len(shapes_list) == 0:
        raise ValueError("shapes_list cannot be empty")
    
    if not isinstance(metadata_df, pd.DataFrame):
        raise ValueError("metadata_df must be a pandas.DataFrame")
    
    if len(metadata_df) == 0:
        raise ValueError("metadata_df cannot be empty")
    
    if len(shapes_list) != len(metadata_df):
        raise ValueError("shapes_list and metadata_df must have the same length")


    # Validate that all items in polygons_list are Polygon objects
    for i in range(len(shapes_list)):
        shape = shapes_list[i]
        if isinstance(shape, list):
            polygon = Polygon(shape)
            shapes_list[i] = polygon

    polygon_indices = list()
    for i, shape in enumerate(shapes_list):
        if isinstance(shape, (Polygon, MultiPolygon)):
            polygon_indices.append(i)
        else:
            warnings.warn(f"Item {i} in shapes_list contains {type(shape)}, ignored", UserWarning)

    polygon_list = [shapes_list[i] for i in polygon_indices]
    metadata_df = metadata_df.iloc[polygon_indices]

    # Create GeoDataFrame with the polygon geometries
    gdf = pd.DataFrame(index=range(len(polygon_list)))
    gdf['geometry'] = polygon_list
    
    # Create nested properties
    classification_properties = []
    for idx in range(len(polygon_list)):
        classification_prop = {}
        
        # Add classification as nested property if specified
        if classification_col is not None:
            if classification_col not in metadata_df.columns:
                raise ValueError(f"Column '{classification_col}' not found in DataFrame")
            classification_name = metadata_df.iloc[idx][classification_col]
            classification_prop['name'] = classification_name
            
        # Add color to classification if color_col is specified
        if color_col is not None:
            if color_col not in metadata_df.columns:
                raise ValueError(f"Column '{color_col}' not found in DataFrame")
            color_value = metadata_df.iloc[idx][color_col]
            classification_prop['color'] = color_value
        
        classification_properties.append(classification_prop)
    
    # Add properties to GeoDataFrame
    gdf['classification'] = classification_properties
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    # Export to GeoJSON
    gdf.to_file(output_file, driver='GeoJSON')

def polygon_to_mask(shapes_list, metadata_df, classification_col=None, image_width=None, image_height=None, categorical_map = None):
    """
    Convert a list of polygon geometries with metadata DataFrame to an integer mask image.
    
    Parameters:
        shapes_list (list): List of shapely.geometry.Polygon objects.
        metadata_df (pd.DataFrame): DataFrame containing metadata for the polygons.
        output_file (str): Path to save the mask image file.
        classification_col (str, optional): Name of the column containing classification strings (e.g., 'cell_type').
    """
    import pandas as pd
    import cv2 as cv
    from shapely.geometry import Polygon, MultiPolygon
    from PIL import Image
    import numpy as np
    import warnings

    # Validate inputs (same sanity checks as polygon_to_geojson)
    if not isinstance(shapes_list, list):
        raise ValueError("polygons_list must be a list")
    
    if len(shapes_list) == 0:
        raise ValueError("polygons_list cannot be empty")
    
    if not isinstance(metadata_df, pd.DataFrame):
        raise ValueError("metadata_df must be a pandas.DataFrame")
    
    if len(metadata_df) == 0:
        raise ValueError("metadata_df cannot be empty")
    
    if len(shapes_list) != len(metadata_df):
        raise ValueError("polygons_list and metadata_df must have the same length")

    # Validate that all items in polygons_list are Polygon objects
    for i in range(len(shapes_list)):
        shape = shapes_list[i]
        if isinstance(shape, list):
            polygon = Polygon(shape)
            shapes_list[i] = polygon

    polygon_indices = list()
    for i, polygon in enumerate(shapes_list):
        if isinstance(polygon, (Polygon, MultiPolygon)):
            polygon_indices.append(i)
        else:
            warnings.warn(f"Item {i} in shapes_list contains {type(shape)}, ignored", UserWarning)
    
    polygons_list = [shapes_list[i] for i in polygon_indices]
    metadata_df = metadata_df.iloc[polygon_indices]

    # Ensure classification column is categorical
    if classification_col is not None:
        if classification_col not in metadata_df.columns:
            raise ValueError(f"Column '{classification_col}' not found in DataFrame")
        
        # Convert classification column to categorical
        if categorical_map is not None:
            cat_codes = [categorical_map[cat] for cat in metadata_df[classification_col].values]
        else:
            assert pd.api.types.is_string_dtype(metadata_df[classification_col]) or pd.api.types.is_categorical_dtype(metadata_df[classification_col])
            if pd.api.types.is_string_dtype(metadata_df[classification_col]):
                metadata_df[classification_col] = pd.Categorical(metadata_df[classification_col])
            cat_codes = list(metadata_df[classification_col].cat.codes)
        
    else:
        # If no classification column, use sequential integers starting from 1
        cat_codes = list(range(len(metadata_df)))

    # Calculate bounding box to determine image dimensions
    all_bounds = [polygon.bounds for polygon in polygons_list]
    minx = min(bounds[0] for bounds in all_bounds)
    miny = min(bounds[1] for bounds in all_bounds)
    maxx = max(bounds[2] for bounds in all_bounds)
    maxy = max(bounds[3] for bounds in all_bounds)
    
    # Add small padding and ensure integer dimensions
    padding = 10
    if image_width is None:
        image_width = int(maxx - minx + 2 * padding)

    if image_height is None:
        image_height = int(maxy - miny + 2 * padding)
    
    # Initialize mask with zeros (default value)
    mask = np.zeros((image_height, image_width), dtype=np.int32)
    
    # Fill polygons with their classification values
    for i, polygon in enumerate(polygons_list):
        # Get the integer value for this polygon (1-based indexing)
        if classification_col is not None:
            mask_value = cat_codes[i] + 1
        else:
            mask_value = i + 1
        
        # Convert polygon coordinates to image coordinates
        if isinstance(polygon, MultiPolygon):
            # Handle MultiPolygon by processing each constituent polygon
            for geom in polygon.geoms:
                coords = np.array(geom.exterior.coords)
                # Adjust coordinates to image space
                coords[:, 0] = coords[:, 0] 
                coords[:, 1] = coords[:, 1] 
                if coords.shape[0] == 0:
                    continue

                cv.fillPoly(mask, [coords.astype('int32')], color=mask_value)
        else:
            # Handle single Polygon
            coords = np.array(polygon.exterior.coords)
            # Adjust coordinates to image space
            coords[:, 0] = coords[:, 0] 
            coords[:, 1] = coords[:, 1]
            if coords.shape[0] == 0:
                continue

            cv.fillPoly(mask, [coords.astype('int32')], color=mask_value)
    
    
    # Return the category mapping for reference
    return mask

def cell_edge_dataframe_to_geojson(df, output_file, start_x_col='start_x', start_y_col='start_y', 
                                  end_x_col='end_x', end_y_col='end_y', classification_col=None, 
                                  color_col=None):
    """
    Convert a DataFrame with start and end coordinates to a GeoJSON file with line segments.

    Parameters:
        df (pd.DataFrame): Input DataFrame with start and end coordinates.
        start_x_col (str): Name of the column containing start x coordinates.
        start_y_col (str): Name of the column containing start y coordinates.
        end_x_col (str): Name of the column containing end x coordinates.
        end_y_col (str): Name of the column containing end y coordinates.
        classification_col (str, optional): Name of the column containing classification strings (e.g., 'cell_type').
        color_col (str, optional): Name of the column containing hex color strings (e.g., '#FF0000').
        output_file (str): Path to save the GeoJSON file.
    """
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import LineString

    # Validate required columns
    required_cols = [start_x_col, start_y_col, end_x_col, end_y_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Create geometry from start and end coordinates
    gdf = pd.DataFrame(index=df.index)
    
    gdf['geometry'] = [LineString([(start_x, start_y), (end_x, end_y)]) 
                      for start_x, start_y, end_x, end_y in 
                      zip(df[start_x_col], df[start_y_col], df[end_x_col], df[end_y_col])]
    
    # Create nested properties
    classification_properties = []
    for idx in df.index:
        classification_prop = {}
        
        # Add classification as nested property if specified
        if classification_col is not None:
            if classification_col not in df.columns:
                raise ValueError(f"Column '{classification_col}' not found in DataFrame")
            classification_name = df.loc[idx, classification_col]
            classification_prop['name'] = classification_name
            
            # Add color to classification if color_col is specified
        if color_col is not None:
            if color_col not in df.columns:
                raise ValueError(f"Column '{color_col}' not found in DataFrame")
            color_value = df.loc[idx, color_col]
            classification_prop['color'] = color_value
        
        classification_properties.append(classification_prop)
    
    # Add properties to GeoDataFrame
    gdf['classification'] = classification_properties
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    # Export to GeoJSON
    gdf.to_file(output_file, driver='GeoJSON')

if __name__ == "__main__":
    from PIL import Image
    qpjson = QuPathGeojson()
    qpjson.read_geojson("/mnt/radonc-li02/public/he_codex/raw/tma/project/external_data/stardist_qupath_prediction-tma2_up_18.HE.geojson")
    image_width, image_height = 3728, 3728
    imask = qpjson.get_integer_mask(image_width, image_height)
    img = Image.fromarray(imask, mode='I')
    img.save("/mnt/radonc-li02/public/he_codex/raw/tma/project/external_data/stardist_qupath_prediction_mask-tma2_up_18.HE.tif")