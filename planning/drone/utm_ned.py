import utm
import numpy as np


def global_to_local(global_position, global_home):
    """
    -> Geodetic(longitude, latitude, altitude)
    -> UTM(east, north, zone_number, zone_letter) = utm.from_latlon(latitude, longitude)
    -> NED(north, east, down)
    """
    (home_east, home_north, _, _) = utm.from_latlon(global_home[1], global_home[0])
    (east, north, _, _) = utm.from_latlon(global_position[1], global_position[0])
    local_position = np.array([north - home_north, east - home_east, -global_position[2]])
    return local_position


def local_to_global(local_position, global_home):
    """
    -> NED(north, east, down)
    -> UTM(lat, long) = utm.to_latlon(east, north, zone_number, zone_letter)
    -> Geodetic(longitude, latitude, altitude)
    """
    (home_east, home_north, zone_number, zone_letter) = utm.from_latlon(global_home[1], global_home[0])
    (lat, lon) = utm.to_latlon(home_east + local_position[1],
                               home_north + local_position[0],
                               zone_number, zone_letter)
    global_position = np.array([lon, lat, -local_position[2]])
    return global_position


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    geodetic_current_coordinates = [-122.079465, 37.393037, 30]
    geodetic_home_coordinates = [-122.108432, 37.400154, 0]
    local_coordinates_NED = global_to_local(geodetic_current_coordinates, geodetic_home_coordinates)
    print(local_coordinates_NED)

    np.set_printoptions(precision=6)
    NED_coordinates = [25.21, 128.07, -30.]
    geodetic_current_coordinates = local_to_global(NED_coordinates, geodetic_home_coordinates)
    print(geodetic_current_coordinates)
