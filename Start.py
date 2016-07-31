# -*- coding: utf-8 -*-
import settings
import telebot
import requests
import urllib
import os

bot = telebot.TeleBot(settings.token)


# TODO возвращение музыки в удобном для проигрывания формате - Саша К
# TODO классификация песен по запросу - Саша З
# TODO перевод классификации песен на py3 - Саша З
# TODO получение нормального текста из гепозиции - ?
# TODO ? обновление классификатора из предпочтений
# TODO ? возвращение вики инфы об объекте
# TODO
#@bot.message_handler(func=lambda message: True, content_types=['text'])
#def check_answer(message):
#    print (message.text)


@bot.message_handler(content_types=["location", "venue"])
def repeat_location_messages(message):
    print(message.location)
    message.location.longitude = message.location.longitude +0.0001;
    message.location.latitude = message.location.latitude + 0.0001;
    bot.send_location(message.chat.id ,  message.location.latitude, message.location.longitude)

@bot.message_handler(content_types=["text" ])
def repeat_text_messages(message): # Название функции не играет никакой роли, в принципе
    #bot.send_chat_action(message.chat.id, 'record_audio')

    r = requests.post("http://muzis.ru/api/stream_from_lyrics.api", data={'lyrics': message.text + ':100', 'size': 1})
    print(r.status_code, r.reason)

    t2 = r.json()
    songs = t2.get('songs')
    for song in songs:

        markup = telebot.types.InlineKeyboardMarkup()

        markup.add(telebot.types.InlineKeyboardButton("Next wav",callback_data= 'next'))
        markup.add(telebot.types.InlineKeyboardButton("Done", callback_data='done'))

        #markup = telebot.types.ReplyKeyboardHide()

        bot.send_message(message.chat.id, song.get('performer') +  song.get('track_name') + '\r\n\r\n')
        ##bot.send_message(message.chat.id, song.get('lyrics'))

        fileMP3 = 'music/'+ str(message.chat.id) + '.mp3'
        f = open(fileMP3, 'wb')
        f.write(urllib.request.urlopen('http://f.muzis.ru/' + song.get('file_mp3')).read())
        f.close()
        bot.send_chat_action(message.chat.id, 'upload_audio')

        audio = open(fileMP3, 'rb')
        print(audio)

        temp = bot.send_audio(message.chat.id, audio, timeout = 1000, title= song.get('performer') +  song.get('track_name'), reply_markup = markup ) # reply_to_message_id=message.message_id) #, reply_markup=markup)
        audio.close()

@bot.message_handler(content_types=[ "photo" ])
def repeat_photo_messages(message): # Название функции не играет никакой роли, в принципе
    temp = bot.send_chat_action(message.chat.id, "upload_photo")
    photoI = None
    maxsize= -1
    for photo  in message.photo:
        if photo.file_size > maxsize :
            maxsize = photo.file_size
            photoI = photo

    file_info = bot.get_file(photoI.file_id)
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(settings.token, file_info.file_path))
    f = open("photo/"  + str(message.chat.id) + ".jpg", 'wb')
    f.write(file.content)
    f.close()
    # TODO API GOOGLE


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    # Если сообщение из чата с ботом
    if call.message:
        if call.data == "next":
            markup = telebot.types.InlineKeyboardMarkup()

            markup.add(telebot.types.InlineKeyboardButton("Next wav", callback_data='next'))
            markup.add(telebot.types.InlineKeyboardButton("Done", callback_data='done'))
            r = requests.post("http://muzis.ru/api/stream_from_lyrics.api",
                              data={'lyrics': 'test' + ':100', 'size': 1})
            print(r.status_code, r.reason)
            t2 = r.json()
            songs = t2.get('songs')
            song = songs[0]
            fileMP3 = 'music/'+ str(call.message.chat.id) + '.mp3'
            f = open(fileMP3, 'wb')
            f.write(urllib.request.urlopen('http://f.muzis.ru/' + song.get('file_mp3')).read())
            f.close()
            bot.send_chat_action(call.message.chat.id, 'upload_audio')

            audio = open(fileMP3, 'rb')
            print(audio)

            #bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text=call.message.text)
            temp = bot.send_voice(call.message.chat.id, audio, timeout = 1000, reply_markup = markup ) # reply_to_message_id=message.message_id) #, reply_markup=markup)
            audio.close()
        else:
            print(1)
            #bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text=call.message.text)
            #bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="Пыщь", reply_markup=markup)
    # Если сообщение из инлайн-режима
    #elif call.inline_message_id:
    #    if call.data == "test":
    #        bot.edit_message_text(inline_message_id=call.inline_message_id, text="Бдыщь")

if __name__ == '__main__':

    if os.path.exists('photo') == False:
        os.mkdir('photo')
    if ~os.path.exists('music') == False:
        os.mkdirs('music')
    bot.polling(none_stop=True)