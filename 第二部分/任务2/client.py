#!/usr/bin/env python3
import sys, rospy
from city_query.srv import CityQuery

def client(city):
    rospy.wait_for_service('query_city')
    resp = rospy.ServiceProxy('query_city', CityQuery)(city)
    print(f"👉 城市: {city} | 邮编: {resp.zip_code} | 位置: {resp.province}")

if __name__ == "__main__":
    client(sys.argv[1])