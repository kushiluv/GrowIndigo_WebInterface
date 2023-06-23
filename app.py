import uuid
import requests
from flask import Flask, render_template, session, request, redirect, url_for
from flask_session import Session 
from flask import jsonify
import msal
import app_config
import json
import time
import pandas as pd
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
app = Flask(__name__)
df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')
df = df.where(pd.notnull(df), None)

            # Convert the DataFrame to a list of dictionaries
df = df.fillna('')
data = df.to_dict(orient='records')

            # Update the coordinates column
for row in data:
                coordinates = row['Coordinates'].split(',')
                points = []
                for coordinate in coordinates:
                    lat_lon_pair = coordinate.split('-')
                    if len(lat_lon_pair) == 2:
                        lat, lon = lat_lon_pair
                        point = {"Latitude": float(lat), "Longitude": float(lon)}
                        points.append(point)
                row['Points'] = points

                # Remove leading and trailing non-breaking space (\xa0) from District
                row['District'] = row['District'].strip('\xa0')

                # Convert date to string with the desired format
                if isinstance(row['Date'], pd.Timestamp):
                    row['Date'] = row['Date'].strftime('%m-%d-%Y')

                coordinates = row['Points']

                # # Compute the area using the coordinates
                # if coordinates:
                #     try:
                #         area = calculate_area(coordinates)
                #         area = area / 4046.856
                #         row['Area'] = area
                #     except ValueError:
                #         print("Invalid coordinates for row:", row)
df['Area'] = [row['Area'] for row in data]






def gpd_geographic_area(geodf):
    if not geodf.crs or not geodf.crs.is_geographic:
        raise TypeError('geodataframe should have a geographic coordinate system')

    geod = geodf.crs.get_geod()
    
    def area_calc(geom):
        if geom.geom_type not in ['MultiPolygon', 'Polygon']:
            return np.nan
        

        if geom.geom_type == 'MultiPolygon':
            return np.sum([area_calc(p) for p in geom.geoms])

        return geod.geometry_area_perimeter(orient(geom, 1))[0]
    
    return geodf.geometry.apply(area_calc)

def calculate_area(points):
 
    geodf = gpd.GeoDataFrame(geometry=[Polygon([(point['Longitude'], point['Latitude']) for point in points])])

    geodf.crs = "EPSG:4326"
    area = gpd_geographic_area(geodf).values[0]

    return area

@app.route('/')
def index1():
    global data
    data_json = json.dumps(data)  # Convert data to JSON string
    items_per_page = 50  # Define the default value of items_per_page

    return render_template('index1.html', data=data_json, items_per_page=items_per_page,page = 1)


@app.route('/filter',methods=['GET', 'POST'])
def filter_polygons():
    global data
    filtered_data =data.copy() 
    page = int(request.args.get('page', 1))
    print("page : ",page)
    items_per_page = 50

    start_index = (page - 1) * items_per_page
    end_index = start_index + items_per_page
    state = request.form.get('state')
    district = request.form.get('district')
    mdo_id = request.form.get('mdo_id')
    search_query = request.form.get('search_query')
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')

    

    if state and state != 'All':
        filtered_data = [entry for entry in filtered_data if entry['State'] == state]

    if district and district != 'All':
        filtered_data = [entry for entry in filtered_data if entry['District'] == district]

    if mdo_id and mdo_id != 'All':
        filtered_data = [entry for entry in filtered_data if str(entry['MDO_ID']) == str(mdo_id)]

    if search_query:
        search_results = []
        for entry in filtered_data:
            # Iterate through each column in the entry
            for column in entry.values():
                # Check if the search query matches the start of any column value
                if str(column).lower().startswith(search_query.lower()):
                    search_results.append(entry)
                    break  # Break the inner loop if a match is found in any column

        filtered_data = search_results

    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        filtered_data = [entry for entry in filtered_data if start_date <= datetime.strptime(entry['Date'], '%m-%d-%Y').date() <= end_date]


    center_lat, center_lng, zoom = calculate_center_coordinates(filtered_data, state, district, mdo_id)
    

    filtered_data = filtered_data[start_index:end_index]
    filtered_data_json = json.dumps(filtered_data)  
    
    return render_template('index1.html', data=filtered_data_json, center_lat=center_lat, center_lng=center_lng, zoom=zoom,items_per_page=items_per_page, page=page)


import csv
@app.route('/save_validation', methods=['POST'])
def save_polygon_validation():
    global data
    print(data[0]['polygon_validation'], data[0]['Farmer_ID'])
    farmer_id = request.form.get('farmer_id')
    field_id = request.form.get('field_id')
    mdo_id = request.form.get('mdo_id')
    validation = request.form.get('validation')
    remark = request.form.get('remark')

    # Update the corresponding row in the data dictionary with the new validation value and remark
    for entry in data:
        if str(entry['Farmer_ID']) == str(farmer_id) and str(entry['Field ID']) == str(field_id):
            entry['polygon_validation'] = validation
            entry['validation_remark'] = remark

    # Save the updated data dictionary as a CSV file
    print("here")
    filename = 'Kushiluv- Polygon data(internship).csv'
    save_data_as_csv(filename, data)
    print("dhere")
    print(data[0]['polygon_validation'])
    # Convert the data dictionary to JSON, replacing NaN values with empty strings
    json_data = pd.DataFrame(data).fillna('').to_json(orient='records')

    # Return the JSON response
    return jsonify({'data': json_data, 'status': 'success'})

def save_data_as_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    global data
    print(data[3]['Coordinates'] , data[0]['Farmer_ID'])
    request_data = request.json
    coordinates = request_data['coordinates']
    farmer_id = request_data['farmer_id']
    field_id = request_data['field_id']
    print("Save -")
    
    # Create coordinates string in the desired format
    coordinates_str = ','.join([f"{coordinate['lat']}-{coordinate['lng']}" for coordinate in coordinates])
    
    # Calculate the area using the coordinates
    points = []
    for coordinate in coordinates:
        lat = coordinate['lat']
        lng = coordinate['lng']
        point = {"Longitude": float(lng), "Latitude": float(lat)}
        points.append(point)
    area = calculate_area(points)
    area = area / 4046.856

    # Update the coordinates and area for the specified farmer_id and field_id in the data dictionary
    for entry in data:
        if str(entry['Farmer_ID']) == str(farmer_id) and str(entry['Field ID']) == str(field_id):
            entry['Coordinates'] = coordinates_str
            entry['Area'] = area

    

    # Save the updated data dictionary back to the Excel file using DataFrame
    df = pd.DataFrame(data.copy())
    with pd.ExcelWriter('Kushiluv- Polygon data(internship).xlsx') as writer:
        df.to_excel(writer, index=False)
    print(data[3]['Coordinates'] , data[0]['Farmer_ID'])
    # Check if the coordinates and area were successfully saved
    saved_coordinates = df.loc[(df['Farmer_ID'] == farmer_id) & (df['Field ID'] == field_id), 'Coordinates'].values[0]
    saved_area = df.loc[(df['Farmer_ID'] == farmer_id) & (df['Field ID'] == field_id), 'Area'].values[0]
    if saved_coordinates == coordinates_str and saved_area == area:
        prompt = 'Coordinates and area have been successfully saved.'
    else:
        prompt = 'Failed to save coordinates and area.'
    json_data = df.fillna('').to_json(orient='records')
    return {'data': json_data, 'status': 'success', 'prompt': prompt}

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
    app.run()
