https://www.dronkert.net/rpi/radio.html

amixer cset numid=3 1
amixer sset PCM 100%


import pygame
pygame.mixer.init()
pygame.mixer.music.load("myFile.wav")
pygame.mixer.music.play()
