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
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
app = Flask(__name__)
app.config.from_object(app_config)
Session(app)

# This section is needed for url_for("foo", _external=True) to automatically
# generate http scheme when this sample is running on localhost,
# and to generate https scheme when it is deployed behind reversed proxy.
# See also https://flask.palletsprojects.com/en/1.0.x/deploying/wsgi-standalone/#proxy-setups
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

@app.route("/")
def index():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template('index.html', user=session["user"], version=msal.__version__)

@app.route("/login")
def login():
    
    session["flow"] = _build_auth_code_flow(scopes=app_config.SCOPE)
    return render_template("login.html", auth_url=session["flow"]["auth_uri"], version=msal.__version__)

@app.route("/getAToken")  # Its absolute URL must match your app's redirect_uri set in AAD
def authorized():
    try:
        cache = _load_cache()
        result = _build_msal_app(cache=cache).acquire_token_by_auth_code_flow(
            session.get("flow", {}), request.args)
        if "error" in result:
            return render_template("auth_error.html", result=result)
        session["user"] = result.get("id_token_claims")
        _save_cache(cache)
        
    except ValueError:  # Usually caused by CSRF
        pass  # Simply ignore them
    return redirect(url_for('index1'))

@app.route("/logout")
def logout():
    session.clear()  # Wipe out user and its token cache from session
    return redirect(  # Also logout from your tenant's web session
        app_config.AUTHORITY + "/oauth2/v2.0/logout" +
        "?post_logout_redirect_uri=" + url_for("index", _external=True))

@app.route("/graphcall")
def graphcall():
    token = _get_token_from_cache(app_config.SCOPE)
    if not token:
        return redirect(url_for("login"))
    graph_data = requests.get(  # Use token to call downstream service
        app_config.ENDPOINT,
        headers={'Authorization': 'Bearer ' + token['access_token']},
        ).json()
    return render_template('display.html', result=graph_data)


def _load_cache():
    cache = msal.SerializableTokenCache()
    if session.get("token_cache"):
        cache.deserialize(session["token_cache"])
    return cache

def _save_cache(cache):
    if cache.has_state_changed:
        session["token_cache"] = cache.serialize()

def _build_msal_app(cache=None, authority=None):
    return msal.ConfidentialClientApplication(
        app_config.CLIENT_ID, authority=authority or app_config.AUTHORITY,
        client_credential=app_config.CLIENT_SECRET, token_cache=cache)

def _build_auth_code_flow(authority=None, scopes=None):
    return _build_msal_app(authority=authority).initiate_auth_code_flow(
        scopes or [],
        redirect_uri=url_for("authorized", _external=True))

def _get_token_from_cache(scope=None):
    cache = _load_cache()  # This web app maintains one cache per session
    cca = _build_msal_app(cache=cache)
    accounts = cca.get_accounts()
    if accounts:  # So all account(s) belong to the current signed-in user
        result = cca.acquire_token_silent(scope, account=accounts[0])
        _save_cache(cache)
        return result

app.jinja_env.globals.update(_build_auth_code_flow=_build_auth_code_flow)  # Used in template


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

from datetime import datetime

def read_excel_data():
    # Read the Excel sheet into a DataFrame
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

        # Convert Timestamp to string
        if isinstance(row['Date'], pd.Timestamp):
            row['Date'] = row['Date'].strftime('%Y-%m-%d %H:%M:%S')

        coordinates = row['Points']

        # Compute the area using the coordinates
        if coordinates:
            try:
                area = calculate_area(coordinates)
                area = area / 4046.856
                row['Area'] = area
            except ValueError:
                print("Invalid coordinates for row:", row)

    df['Area'] = [row['Area'] for row in data]
    print(type(df['Date'][0]))
    with pd.ExcelWriter('Kushiluv- Polygon data(internship).xlsx') as writer:
        df.to_excel(writer, index=False)

    return data


@app.route('/index1')
def index1():
    if not session.get("user"):
        return redirect(url_for("login"))
    
    user = session["user"]  # Get the authenticated user from the session
    data = read_excel_data()
    data_json = json.dumps(data)  # Convert data to JSON string
    return render_template('index1.html', data=data_json)

@app.route('/redirect_index1', methods=['GET', 'POST'])
def redirect_index1():
    print("hjere")
    time.sleep(1)  # Delay the redirect for 1 second
    return redirect(url_for('index1'))

from datetime import datetime


@app.route('/filter', methods=['POST'])
def filter_polygons():
    state = request.form.get('state')
    district = request.form.get('district')
    mdo_id = request.form.get('mdo_id')
    search_query = request.form.get('search_query')
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')

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

    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        data = [entry for entry in data if start_date <= datetime.strptime(entry['Date'], '%Y-%m-%d %H:%M:%S') <= end_date]


    center_lat, center_lng, zoom = calculate_center_coordinates(data, state, district, mdo_id)

    data_json = json.dumps(data)  # Convert filtered data to JSON string

    return render_template('index1.html', data=data_json, center_lat=center_lat, center_lng=center_lng, zoom=zoom)


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
    print("Save -")
    
    # Create coordinates string in the desired format
    coordinates_str = ','.join([f"{coordinate['lat']}-{coordinate['lng']}" for coordinate in coordinates])
    print(coordinates_str)
    # Calculate the area using the coordinates
    points = []
    for coordinate in coordinates:
        lat = coordinate['lat']
        lng = coordinate['lng']
        point = {"Longitude": float(lng), "Latitude": float(lat)}
        points.append(point)
    area = calculate_area(points)
    area = area / 4046.856

    # Read the Excel file
    df = pd.read_excel('Kushiluv- Polygon data(internship).xlsx')

    # Update the coordinates and area for the specified farmer_id and field_id
    condition = (df['Farmer_ID'] == farmer_id) & (df['Field ID'] == field_id)
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
    app.run()

