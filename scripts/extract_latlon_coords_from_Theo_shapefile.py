import geopandas as gpd
df = gpd.read_file('PerimEtude_06.shp')
df_latlon=df.to_crs(epsg='4326')
df_latlon.get_coordinates()

