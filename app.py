from flask import Flask,jsonify, render_template, url_for, request
import json
import pandas as pd
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
app = Flask(__name__)

def gpd_geographic_area(geodf):
    if not geodf.crs or not geodf.crs.is_geographic:
        raise TypeError('geodataframe should have a geographic coordinate system')

    geod = geodf.crs.get_geod()
    
    def area_calc(geom):
        if geom.geom_type not in ['MultiPolygon', 'Polygon']:
            return np.nan
        
        # For MultiPolygon, calculate area for each polygon separately
        if geom.geom_type == 'MultiPolygon':
            return np.sum([area_calc(p) for p in geom.geoms])

        # Orient to ensure a counter-clockwise traversal. 
        # See https://pyproj4.github.io/pyproj/stable/api/geod.html
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(orient(geom, 1))[0]
    
    return geodf.geometry.apply(area_calc)

def calculate_area(points):
 
    geodf = gpd.GeoDataFrame(geometry=[Polygon([(point['Longitude'], point['Latitude']) for point in points])])

    geodf.crs = "EPSG:4326"
    area = gpd_geographic_area(geodf).values[0]

    return area

def read_excel_data():
    # Read the Excel sheet into a DataFrame
    df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')
    df = df.where(pd.notnull(df), None)
    
    # Convert the DataFrame to a list of dictionaries
    df = df.fillna('')
    data = df.to_dict(orient='records')

    # Update the coordinates column
    for row in data:
        coordinates = row['Coordinates'].split(' ')
        points = []
        for coordinate in coordinates:
            values = coordinate.split(',')
            if len(values) >= 2:
                lon, lat = values[:2]  # Extract first two values
                point = {"Latitude": float(lat), "Longitude": float(lon)}
                points.append(point)
        row['Points'] = points
        
        del row['Coordinates']

        # Remove leading and trailing non-breaking space (\xa0) from District
        row['District'] = row['District'].strip('\xa0')
        coordinates = row['Points']
        
        # Compute the area using the coordinates
        if coordinates:
            try:
                area = calculate_area(coordinates)
                area = area/4046.856
                row['Area'] = area
                
                print(row['Farmer_ID'], "Area:", area)
            except ValueError:
                print("Invalid coordinates for row:", row)
    
    df['Area'] = [row['Area'] for row in data]
    
    with pd.ExcelWriter('Kushiluv- Polygon data(internship).xlsx') as writer:
        df.to_excel(writer, index=False)
    
    return data

@app.route('/')
def index():
    data = read_excel_data()
    data_json = json.dumps(data)  # Convert data to JSON string
    return render_template('index.html', data=data_json)

@app.route('/filter', methods=['POST'])
def filter_polygons():
    state = request.form.get('state')
    district = request.form.get('district')
    mdo_id = request.form.get('mdo_id')
    search_query = request.form.get('search_query')
    data = read_excel_data()
    
    if state and state != 'All':
        data = [entry for entry in data if entry['State'] == state]

    if district and district != 'All':
        data = [entry for entry in data if entry['District'] == district]
    
    if mdo_id and mdo_id != 'All':
        data = [entry for entry in data if str(entry['MDO_ID']) == str(mdo_id)]
    
    if search_query:
        search_results = []
        for entry in data:
            # Iterate through each column in the entry
            for column in entry.values():
                # Check if the search query matches the start of any column value
                if str(column).lower().startswith(search_query.lower()):
                    search_results.append(entry)
                    break  # Break the inner loop if a match is found in any column
    
        data = search_results
    
    center_lat, center_lng ,zoom = calculate_center_coordinates(data, state, district, mdo_id)
    
    data_json = json.dumps(data)  # Convert filtered data to JSON string
    
    return render_template('index.html', data=data_json, center_lat=center_lat, center_lng=center_lng, zoom=zoom)

@app.route('/save_validation', methods=['POST'])
def save_polygon_validation():
    farmer_id = request.form.get('farmer_id')
    field_id = request.form.get('field_id')
    mdo_id = request.form.get('mdo_id')
    validation = request.form.get('validation')
    remark = request.form.get('remark')

    # Load the Excel file into a DataFrame
    df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')

    # Check if the "polygon_validation" column already exists in the DataFrame
    if 'polygon_validation' not in df.columns:
        # If it doesn't exist, create the column and initialize it with empty values
        df['polygon_validation'] = ''

    # Check if the "validation_remark" column already exists in the DataFrame
    if 'validation_remark' not in df.columns:
        # If it doesn't exist, create the column and initialize it with empty values
        df['validation_remark'] = ''

    # Convert the values in the "Farmer_ID" and "Field ID" columns to strings
    df['Farmer_ID'] = df['Farmer_ID'].astype(str)
    df['Field ID'] = df['Field ID'].astype(str)

    # Remove commas from the "Farmer_ID" column values
    df['Farmer_ID'] = df['Farmer_ID'].str.replace(',', '')

    # Replace NaN values with empty strings in the DataFrame
    df = df.replace(np.nan, '', regex=True)

    # Update the corresponding row with the new validation value and remark
    mask = (df['Farmer_ID'] == farmer_id) & (df['Field ID'] == field_id)
    df.loc[mask, 'polygon_validation'] = validation
    df.loc[mask, 'validation_remark'] = remark

    # Save the updated DataFrame back to the Excel file using ExcelWriter
    with pd.ExcelWriter('Kushiluv- Polygon data(internship).xlsx') as writer:
        df.to_excel(writer, index=False)

    # Convert the DataFrame to JSON, replacing NaN values with empty strings
    json_data = df.fillna('').to_json(orient='records')

    # Return the JSON response
    return jsonify({'data': json_data, 'status': 'success'})
@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    data = request.json
    coordinates = data['coordinates']
    farmer_id = data['farmer_id']
    field_id = data['field_id']

    # Create points from the coordinates
    points = []
    for coordinate in coordinates:
        print(coordinate)
        lat = coordinate['lat']
        lng = coordinate['lng']
        point = {"Longitude": float(lng), "Latitude": float(lat)}
        points.append(point)

    # Calculate the area using the points
    area = calculate_area(points)
    area = area/4046.856
    # Read the Excel file
    df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')

    # Update the coordinates and area for the specified farmer_id and field_id
    condition = (df['Farmer_ID'] == farmer_id) & (df['Field ID'] == field_id)
    coordinates_str = ' '.join([f"{point['Longitude']},{point['Latitude']},0" for point in points])
    df.loc[condition, 'Coordinates'] = coordinates_str
    df.loc[condition, 'Area'] = area

    # Save the updated DataFrame back to the Excel file
    df.to_excel('Kushiluv- Polygon data(internship).xlsx', index=False)

    # Check if the coordinates and area were successfully saved
    saved_coordinates = df.loc[condition, 'Coordinates'].values[0]
    saved_area = df.loc[condition, 'Area'].values[0]
    if saved_coordinates == coordinates_str and saved_area == area:
        prompt = 'Coordinates and area have been successfully saved.'
    else:
        prompt = 'Failed to save coordinates and area.'

    return {'status': 'success', 'prompt': prompt}
def calculate_center_coordinates(data, state=None, district=None,mdo_id=None):
    
    if district and district != 'All':
        filtered_data = [entry for entry in data if entry['District'] == district]
    elif state and state != 'All':
        filtered_data = [entry for entry in data if entry['State'] == state]
    elif mdo_id and mdo_id != 'All':
        filtered_data = [entry for entry in data if entry['MDO_ID'] == mdo_id]    
    else:
        filtered_data = data
    #print(filtered_data)
    if filtered_data:
        latitudes = [point['Latitude'] for entry in filtered_data for point in entry['Points']]
        longitudes = [point['Longitude'] for entry in filtered_data for point in entry['Points']]
        center_lat = sum(latitudes) / len(latitudes)
        center_lng = sum(longitudes) / len(longitudes)

        # Calculate the zoom level based on the bounding box of the filtered data
        min_lat = min(latitudes)
        max_lat = max(latitudes)
        min_lng = min(longitudes)
        max_lng = max(longitudes)

        lat_diff = max_lat - min_lat
        lng_diff = max_lng - min_lng

        # Adjust the zoom level based on the latitude or longitude difference
        zoom_lat = int(round(math.log(360 / lat_diff, 2)))
        zoom_lng = int(round(math.log(360 / lng_diff, 2)))
        zoom = min(zoom_lat, zoom_lng)

        if state!='ALL' and district == 'ALL':
            zoom += 1# If a state is selected, adjust the zoom level for a better view of the state
        elif state=='ALL' and district=='ALL':
            zoom = 5
      
        return center_lat, center_lng, zoom
    else:
        # Default center coordinates and zoom level if no data available
        return 20.5937, 78.9629, 5  # Center of India with a default zoom level of 5



if __name__ == "__main__":
    app.run(debug=True)