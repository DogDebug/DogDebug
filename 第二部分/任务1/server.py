#!/usr/bin/env python3
import rospy
from city_query.srv import CityQuery, CityQueryResponse

DATABASE = {"广州":("510000","广东省"), "成都":("610000","四川省"), "首尔":("04547","韩国"), "悉尼":("2000","澳大利亚"), "纽约":("10001","美国"), "杭州":("310000","浙江省"), "卑尔根":("5003","挪威"), "普洱":("665000","云南省"), "西安":("710000","陕西省"), "景德镇":("333000","江西省")}

def handle_query(req):
    if req.city in DATABASE:
        zip_code, province = DATABASE[req.city]
        rospy.loginfo(f"查询成功: {req.city} -> 邮编:{zip_code}, 位置:{province}")
        return CityQueryResponse(zip_code, province)
    return CityQueryResponse("未知", "未知")

rospy.init_node('city_server')
rospy.Service('query_city', CityQuery, handle_query)
rospy.loginfo("🚀 城市查询服务端已启动，等待请求...")
rospy.spin()