from flask import Flask,jsonify, render_template, url_for, request
import json
import pandas as pd
import math
import numpy as np
app = Flask(__name__)

def read_excel_data():
    # Replace this with your own logic to read the data from Excel
    # Read the Excel sheet into a DataFrame
    df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')
    df = df.where(pd.notnull(df), None)
    # Convert the DataFrame to a list of dictionaries
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

    data = read_excel_data()

    if state and state != 'All':
        data = [entry for entry in data if entry['State'] == state]

    if district and district != 'All':
        data = [entry for entry in data if entry['District'] == district]

    center_lat, center_lng ,zoom = calculate_center_coordinates(data, state, district)

    data_json = json.dumps(data)  # Convert filtered data to JSON string
    
    return render_template('index.html', data=data_json, center_lat=center_lat, center_lng=center_lng, zoom=zoom)


@app.route('/save_validation', methods=['POST'])
def save_polygon_validation():
    farmer_id = request.form.get('farmer_id')
    field_id = request.form.get('field_id')
    mdo_id = request.form.get('mdo_id')
    validation = request.form.get('validation')
    
    # Load the Excel file into a DataFrame
    df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')

    # Check if the "polygon_validation" column already exists in the DataFrame
    if 'polygon_validation' not in df.columns:
        # If it doesn't exist, create the column and initialize it with empty values
        df['polygon_validation'] = ''

    # Convert the values in the "Farmer_ID" and "Field ID" columns to strings
    df['Farmer_ID'] = df['Farmer_ID'].astype(str)
    df['Field ID'] = df['Field ID'].astype(str)

    # Remove commas from the "Farmer_ID" column values
    df['Farmer_ID'] = df['Farmer_ID'].str.replace(',', '')

    # Replace NaN values with empty strings in the DataFrame
    df = df.replace(np.nan, '', regex=True)

    # Update the corresponding row with the new validation value
    mask = (df['Farmer_ID'] == farmer_id) & (df['Field ID'] == field_id)
    df.loc[mask, 'polygon_validation'] = validation

    # Save the updated DataFrame back to the Excel file using ExcelWriter
    with pd.ExcelWriter('Kushiluv- Polygon data(internship).xlsx') as writer:
        df.to_excel(writer, index=False)

    # Convert the DataFrame to JSON, replacing NaN values with empty strings
    json_data = df.fillna('').to_json(orient='records')

    # Return the JSON response
    return jsonify({'data': json_data, 'status': 'success'})

def calculate_center_coordinates(data, state=None, district=None):
    
    if district and district != 'All':
        filtered_data = [entry for entry in data if entry['District'] == district]
    elif state and state != 'All':
        filtered_data = [entry for entry in data if entry['State'] == state]
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