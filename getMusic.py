import requests

index = 0;
#while  True:
r = requests.post("http://muzis.ru/api/search.api", data={'q_value' : 11 })
print(r.status_code, r.reason)
t1 = r.json()
print(t1)
songs = t1.get('songs')

size = len(songs)
#    if size == 0:
#        break
#    index += size
#print (index)
print (songs[0].get('file_mp3') )
print (songs[0].get('lyrics'))

r = requests.post("http://muzis.ru/api/stream_from_obj.api", data={'type':2, 'id': songs[0].get('id')})
print(r.status_code, r.reason)

t1 = r.json()
print(t1)
songs = t1.get('songs')
print (songs[0].get('lyrics'))
    #t = r.text
    #print(message.location)
    ##print (bot.send_chat_action(message.chat.id, "location"))
    #print(t)
t2 = r.json()
print (t2)
#    songs = t2.get('songs')
 #   for song in songs:
  #      bot.send_message(message.chat.id, song.get('track_name') + '\r\n\r\n')
   #     bot.send_message(message.chat.id, song.get('lyrics'))
        # http://f.muzis.ru/
    #    bot.send_message(message.chat.id, song.get('file_mp3'))
        #audio = open('http://f.muzis.ru/' + song.get('file_mp3'), 'rb')
        #bot.send_voice(message.chat.id, 'http://f.muzis.ru/' + song.get('file_mp3')) #, reply_markup=markup)
        #bot.send_audio(message.chat.id, audio) #, reply_markup=markup)
        #bot.send_voice()
    #bot.send_message(message.chat.id, message.text + 'dfdf')
