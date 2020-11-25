# Coorinates transformation

# Imports
from math import cos, radians, sin, sqrt
import pandas as pd
import numpy as np

# Ellipsoid constants, parameters: semi major axis in metres, reciprocal flattening.
GRS80 = 6378137, 298.257222100882711
WGS84 = 6378137, 298.257223563

# Function that calculates geocentric coordinates from geodetic data


def geodetic_to_geocentric(ellipsoid, latitude, longitude, height):
    """Return geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates given by latitude and longitude (in
    degrees) and height above ellipsoid. The ellipsoid must be
    specified by a pair (semi-major axis, reciprocal flattening).
    """
    φ = radians(latitude)
    λ = radians(longitude)
    sin_φ = sin(φ)
    a, rf = ellipsoid           # semi-major axis, reciprocal flattening
    e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
    n = a / sqrt(1 - e2 * sin_φ ** 2)  # prime vertical radius
    r = (n + height) * cos(φ)   # perpendicular distance from z axis
    x = r * cos(λ)
    y = r * sin(λ)
    z = (n * (1 - e2) + height) * sin_φ
    return x, y, z


# Function that retrieves geodetic coordinates from Romania cities
def geoVals():
    coordinates = {}
    coordinates["Zerind"] = geodetic_to_geocentric(
        WGS84, 46.6166700, 21.5166700, 85)
    coordinates["Rimnicu_Vilcea"] = geodetic_to_geocentric(
        WGS84, 45.1000000, 24.3666700, 237)
    coordinates["Timișoara"] = geodetic_to_geocentric(
        WGS84, 45.7537200, 21.2257100, 96)
    coordinates["Targu_Neamt"] = geodetic_to_geocentric(
        WGS84, 47.2000000, 26.3666700, 361)
    coordinates["Pitesti"] = geodetic_to_geocentric(
        WGS84, 44.8500000, 24.8666700, 307)
    coordinates["Urziceni"] = geodetic_to_geocentric(
        WGS84, 44.7166700, 26.6333300, 52)
    coordinates["Fagaras"] = geodetic_to_geocentric(
        WGS84, 45.8500000, 24.9666700, 420)
    coordinates["Oradea"] = geodetic_to_geocentric(
        WGS84, 47.0458, 21.91833, 131)
    coordinates["Sibiu"] = geodetic_to_geocentric(
        WGS84, 45.8000000, 24.1500000, 410)
    coordinates_df = pd.DataFrame(coordinates, index=["x", "y", "z"])
    return coordinates_df

# Function that calculates distances between Romania cities


def distance():
    pdCities = geoVals()
    citiesMatrix = pdCities.to_numpy()
    x = citiesMatrix[0]
    y = citiesMatrix[1]
    z = citiesMatrix[2]
    diffx = np.diff(x)
    diffy = np.diff(y)
    diffz = np.diff(z)
    sumOfSquares = diffx**2 + diffy**2 + diffz**2
    sqrtOfSquares = np.sqrt(sumOfSquares)
    print("Distances between Romania cities (km): ", sqrtOfSquares/1000)
