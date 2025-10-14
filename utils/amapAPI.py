import pandas as pd
import transbigdata
import xlwt
import urllib.request
from bs4 import BeautifulSoup
import re
import os


class getPOI(object):
    def __init__(self, save_path, key, types, city):
        self.save_path = save_path
        self.key = key
        self.types = types
        self.city = city

    def parsePOI(self):
        poiTag = ["id", "name", "type", "typecode", "biz_type", "address", "location", "tel", "pname", "cityname", "adname"]
        poiSoupTag = ["idSoup", "nameSoup", "typeSoup", "typecodeSoup", "biz_typeSoup", "addressSoup", "locationSoup", "telSoup", "pnameSoup", "citynameSoup", "adnameSoup"]
        pattern = re.compile("(?:>)(.*?)(?=<)", re.S)
        poiExcel = xlwt.Workbook()
        sheet = poiExcel.add_sheet("poiResult", cell_overwrite_ok=True)
        for colIndex in range(len(poiTag)):
            sheet.write(0, colIndex, poiTag[colIndex])
        offset = 20
        maxPage = 90

        for pageIndex in range(1, maxPage + 1):
            try:
                url = "http://restapi.amap.com/v3/place/text?&keywords=&types=" + self.types + "&city=" + self.city + "&citylimit=true&output=xml&offset=" + \
                      str(offset) + "&page=" + str(pageIndex) + "&key=" + self.key
                poiSoup = BeautifulSoup(urllib.request.urlopen(url).read(), "xml")
                for tagIndex in range(len(poiTag)):
                    poiSoupTag[tagIndex] = poiSoup.findAll(poiTag[tagIndex])
                if len(poiSoupTag[0]) == 0:
                    break
                for rowIndex in range(len(poiSoupTag[0])):
                    for colIndex in range(len(poiSoupTag)):
                        sheet.write(len(poiSoupTag[0]) * (pageIndex - 1) + rowIndex + 1, colIndex,
                                    re.findall(pattern, str(poiSoupTag[colIndex][rowIndex])))
            except Exception as e:
                print(e)

        poiExcel.save(self.save_path + "/路口poi_" + self.city + ".xls")

    def run_argorithm(self):
        self.parsePOI()

def choice_road(data, node_num):
    data['lng'] = data['location'].apply(lambda x: x.split(',')[0])
    data['lat'] = data['location'].apply(lambda x: x.split(',')[1])
    # GCJ02 --> WGS08
    data['lng'] = data.apply(lambda x: transbigdata.gcj02towgs84(x['lng'], x['lat'])[0], axis=1)
    data['lat'] = data.apply(lambda x: transbigdata.gcj02towgs84(x['lng'], x['lat'])[1], axis=1)
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=node_num, random_state=10)
    y_pred = model.fit_predict(data[['lng', 'lat']])
    data['lable'] = y_pred
    data = data.drop_duplicates(subset=['lable']).reset_index(drop=True)
    return data


if __name__ == '__main__':
    # 获取高德的路口POI数据
    # save_path = "../data/node_location/"
    # key = "39c2286b9f987ebbf4446be1ba5e5e96" # csj
    # types = "150500"  # 地铁站POI
    # city = '510108'
    # node_location = getPOI(save_path, key, types, city)
    # node_location.run_argorithm()

    # # 聚类获取目标节点经纬度和编码lable
    # os.chdir('../')
    # print(os.getcwd())
    # data = pd.read_excel(r'./data/node_location/CD_road_poi.xlsx')
    # data = choice_road(data=data, node_num=400)
    # data.to_csv(r'../data/node_location/CD_road_400.csv', index=False)

    names = os.listdir('../data/node_location/')
    res = pd.DataFrame()
    for name in names:
        if '路口' in str(name):
            df = pd.read_excel(os.path.join('../data/node_location/', name))
            res = res.append(df)
    res.to_csv('../data/node_location/CD_railway.csv', index=False)



