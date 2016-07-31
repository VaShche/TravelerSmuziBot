# -*- coding: utf-8 -*-
import os
import re

import requests


class MuziGetter(object):

    def __init__(self):
        import muziStyles
        self.muzi_styles = muziStyles.muziStyles
        self.output = r'C:\Hakaton\Projects\TSBsongs'

    def saveData(self, song):
        song_id = song.get('id', '')
        song_text = re.sub(r'[a-zA-Z]+', '', song.get('lyrics', ''))
        if song_text != '':
            with open(os.path.join(self.output, str(song_id)), 'wb') as out_data:
                out_data.write(song_text.encode())

    def savePerformers(self, performers):
        with open(os.path.join(self.output, 'performers.txt'), 'wb') as out_data:
            out_data.writelines(list(map(lambda x: '{}\r\n'.format(x).encode(), performers.values())))
            #out_data.writelines(list(map(lambda x: '{}\t{}'.format(str(x[0]), x[1].encode()), performers.items())))


    def getSongsByStyle(self, style_id):
        r = requests.post("http://muzis.ru/api/search.api",
                          data={'q_value': style_id, 'size': 200})
        j = r.json()
        results = []
        for song in j.get('songs', []):
            results.append(song)
        return results

    def save_all(self):
        performers = {}
        for style in self.muzi_styles.keys():
            for song in self.getSongsByStyle(style):
                performers[song.get('performer_id', 0)] = song.get('performer', '')
                self.saveData(song)
                self.savePerformers(performers)


if __name__ == '__main__':
    m = MuziGetter()
    m.save_all()
