import urllib
import wikipedia
from urllib.request import urlopen
import json

apikey = 'AIzaSyDcIAqGLWazfvs4D5N8b8Y165bEp-g9wPw'


def inf(d, s):
    wikipedia.summary("Wikipedia")
    wikipedia.set_lang("ru")

    types = ['museum', 'park', 'church', 'zoo', 'train_station', 'stadium']

    def param(t):
        content = urlopen(
            'https://maps.googleapis.com/maps/api/place/nearbysearch/json?language=ru&location=' + str(d) + ',' + str(
                s) + '&rankby=distance&types=' + t + '&key=' + apikey).read()
        c = json.loads(content.decode("utf-8"))
        c = c.get('results')
        if len(c) != 0:
            c = c[0].get('name')
            # print(c)
            m = wikipedia.search(c, results=5, suggestion=False)
            # print(m[0])
            if len(m) != 0:
                textsong = wikipedia.summary(m, sentences=5, chars=1)
                if textsong != '':
                    return textsong
            #print(textsong)
            # if len(wikipedia.search(c)) != 0:
            #    st = wikipedia.page(c)
            # if st.content
            #    print(st.content)

    for type in types: #i in range(6):
        temp =  param(type)
        if temp:
            return temp



#print(inf(59.9682258, 30.3215306))
