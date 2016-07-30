# -*- coding: utf-8 -*-
import os

import requests


class MuziGetter(object):

    def __init__(self):
        import muziStyles
        self.muzi_styles = muziStyles.muziStyles
        self.output = r'C:\Hakaton\Projects\TSBsongs'

    def saveData(self, song):
        song_id = song.get('id', '')
        song_text = song.get('lyrics', '')

        if song_text != '':
            with open(os.path.join(self.output, str(song_id)), 'wb') as out_data:
                out_data.write(song_text.encode())


    def getSongsByStyle(self, style_id):
        r = requests.post("http://muzis.ru/api/search.api",
                          data={'q_value': style_id, 'size': 200})
        j = r.json()
        results = []
        for song in j.get('songs', []):
            results.append(song)
        return results

    def save_all(self):
        for style in self.muzi_styles.keys():
            for song in self.getSongsByStyle(style):
                self.saveData(song)


if __name__ == '__main__':
    
    m = MuziGetter()
    m.save_all()
