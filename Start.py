# -*- coding: utf-8 -*-
import settings
import telebot
import requests
bot = telebot.TeleBot(settings.token)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message): # Название функции не играет никакой роли, в принципе
    r = requests.post("http://muzis.ru/api/stream_from_lyrics.api", data={'lyrics': message.text, 'size': 2})
    print(r.status_code, r.reason)
    t = r.text
    print(t)
    t2 = r.json()
    songs = t2.get('songs')
    for song in songs:
        bot.send_message(message.chat.id, song.get('track_name') + '\r\n\r\n')
        bot.send_message(message.chat.id, song.get('lyrics'))
        # http://f.muzis.ru/
        bot.send_message(message.chat.id, song.get('file_mp3'))
        #audio = open('http://f.muzis.ru/' + song.get('file_mp3'), 'rb')
        #bot.send_voice(message.chat.id, 'http://f.muzis.ru/' + song.get('file_mp3')) #, reply_markup=markup)
        #bot.send_audio(message.chat.id, audio) #, reply_markup=markup)
        #bot.send_voice()
    #bot.send_message(message.chat.id, message.text + 'dfdf')

if __name__ == '__main__':

#    print(t2)

    bot.polling(none_stop=True)