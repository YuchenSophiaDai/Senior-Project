import spotipy
import requests
import urllib.request
import re
from bs4 import BeautifulSoup
import spotipy.util as util
import spotipy.oauth2 as oauth2
import pandas as pd
import csv
import sys
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import csvkit
import pickle
import statistics 

c = 'http://www.bbc.co.uk/radio1/essentialmix/tracklistings'
u1 = c + '2009.shtml'
u2 = c + '2008.shtml'
u3 = c + '2007.shtml'
u4 = c + '2006.shtml'
u5 = c + '2005.shtml'
u6 = c + '2004.shtml'
u7 = c + '2003.shtml'
u8 = c + '2002.shtml'

urls = [u1,u2,u3,u4,u5,u6,u7,u8]
main = 'http://www.bbc.co.uk'
source = 'https://tracklists.thomaslaupstad.com/page/'

username = 'yuchen.dai'

auth = oauth2.SpotifyClientCredentials(
    client_id='a9323de006114203a70807a9ff000c2e',
    client_secret = '5a0822a41f044b1cbeacc0460ecfa72f'
)

#token = auth.get_access_token()
#spotify= spotipy.Spotify(auth=token)

def clean_text(s):
	s = s.split(')')[-1]
	s = s.split('[')[0]
	if " ft." in s:
		i = s.find(' ft.')
		s = s[:i] + ' ' + s[i:]
	if " vs." in s:
		i = s.find(' ft')
		s = s[:i] + ' ' + s[i:]
	if "/" in s:
		i = s.find('/')
		s = s[:i] + ' ' + s[i:]
	split = s.strip().split('-')
	if len(split) == 1:
		return split[0]
	else:
		return split[0] + ' ' + split[1]

def clean(s):
	s = s.split('(')[0]
	s = s.split('[')[0]
	if " ft." in s:
		i = s.find(' ft.')
		s = s[:i] + ' ' + s[i:]
	if " vs." in s:
		i = s.find(' ft')
		s = s[:i] + ' ' + s[i:]
	if "/" in s:
		i = s.find('/')
		s = s[:i] + ' ' + s[i:]
	split = s.strip().split('-')
	if len(split) == 1:
		return split[0]
	else:
		return split[0] + '  ' + split[1]

def clean_thomas(s, k):
	s = s[:k] + '' + s[k+1:]
	s = s.split('(')[0]
	s = s.split('[')[0]
	s = s.split(']')[-1]
	if " ft." in s:
		i = s.find(' ft.')
		s = s[:i] + ' ' + s[i:]
	if " vs." in s:
		i = s.find(' ft')
		s = s[:i] + ' ' + s[i:]
	if "/" in s:
		i = s.find('/')
		s = s[:i] + ' ' + s[i:]
	return s

#-*- coding: utf-8 -*-
def playlistify_thomas(s):
	all_playlists = []
	for i in range(880):
		playlist = []
		url = s + str(i)
		try: 
			response = requests.get(url)
			soup = BeautifulSoup(response.text, "html.parser")
			for link in soup.find_all('div',attrs={"class":"entry-content"}):
				t = link.text
				#print(t)
				#break
				lines = t.split('\n')
				for line in lines:
					if 'tracklist' in line:
						if len(playlist) > 5:
							all_playlists.append(playlist)
						playlist = []
						continue
					k = 0
					for i in range(0, len(line)):
						if ord(line[i])>127:
							k = i
							break
					if k!=0:
						song = clean_thomas(line, k)
						playlist.append(song)
		except Exception as e: 
			print(e)
			print(i)
			pass
	return all_playlists

def extract_songs_bbc(urls):
	song_url = set()
	for url in urls:
		response = requests.get(url)
		soup = BeautifulSoup(response.text, "html.parser")

		for link in soup.find_all('div',attrs={"class":"tracklisting_for_month"}):
			tags = link.find_all('a')
			for t in tags:
				u = t.get('href')
				if main not in u:
					song_url.add(main+u)
	return song_url

def playlistify_bbc(urls):
	all_playlists = []
	for u in urls:
		playlist = []
		response = requests.get(u)
		soup = BeautifulSoup(response.text, "html.parser")
		for link in soup.find_all('div',attrs={"class":"tracklisting_archive_container"}):
			t = link.text
			#print(t)
			lines = t.split('\n')
			for line in lines:
				if "Your reviews of the mix" in line:
					break
				if "\'" in line:
					song = clean(line)
					playlist.append(song)
		all_playlists.append(playlist)
	return all_playlists

def playlistify_text(file):
	f = open(file,"r") 
	data = f.readlines() 
	all_playlists = []
	playlist = []
	for line in data:
		read = False
		song = ''
		if line == '\n':
			if len(playlist)>1:
				all_playlists.append(playlist)
			playlist= []
		for c in line:
			if c == '\n':
				song = clean(song)
				playlist.append(song)
				break
			if c == '.':
				read = True
				continue
			if read == True:
				song = song + c
	f.close()
	return all_playlists

def get_track_id(q, spotify):
	try:
		result = spotify.search(q, limit=1, offset=0, type='track', market=None)
		if len(result['tracks']['items'])!= 0:
			track_id = result['tracks']['items'][0]['id']
			return track_id
		else:
			name = q.split('  ')[-1]
			result = spotify.search(name, limit=1, offset=0, type='track', market=None)
			if len(result['tracks']['items'])!= 0:
				track_id = result['tracks']['items'][0]['id']
				return track_id
			else:
				return ''
	except Exception as e:
		print(e) 
		pass
	return -1

def spotify_id(all_playlists):
	#all_playlists_id = []
	count = 0
	token = auth.get_access_token()
	spotify= spotipy.Spotify(auth=token)
	for playlist in all_playlists:	
		playlist = list(filter(None, playlist))
		filtered = len(playlist)
		if filtered < 5:
			continue
		playlist_id = []
		for song in playlist:
			q = song
			if q.strip() == '':
				continue
			#print(q)
			try: 
				track = get_track_id(q, spotify)
				if track == -1:
					token = auth.get_access_token()
					spotify= spotipy.Spotify(auth=token)
					track = get_track_id(q, spotify)
					if track == -1:
						continue
				if track == '':
					break
				playlist_id.append(track)
			except Exception as e: 
				print(e)
				#print(q)
				token = auth.get_access_token()
				spotify= spotipy.Spotify(auth=token)
				track = get_track_id(q, spotify)
				if track == '':
					break
				if track == -1:
					continue
				playlist_id.append(track)
				count = count +1
				pass
		if len(playlist_id) > 4:
			#all_playlists_id.append(playlist_id)
			with open('final_thomas_perfect.csv','a+', newline ='') as f:
				writer = csv.writer(f)
				writer.writerow(playlist_id)
		else:
			print(playlist_id)

def getContents():
	with open('thomas_songs_new.csv', newline='',  encoding='utf-8') as csvfile:
		data = list(csv.reader(csvfile))
	spotify_id(data)


def get_timbre(song_id, spotify):
	try: 
		info = spotify.audio_analysis(song_id)['segments']
		if info == None:
			return None
		curr = np.zeros(12)
		for seg in info:
			vec = seg['timbre']
			zipped_lists = zip(curr, vec)
			curr = [x + y for (x, y) in zipped_lists]
		l = len(info)
		avg = [number/l for number in curr]
		return avg
	except Exception as e:
		print(e) 
		pass
	return None

def get_vector(song_id, spotify):
	try: 
		info = spotify.audio_features(song_id)[0]
		if info!= None:
			vector = [info['danceability'], info['energy'], info['key'], info['loudness'], info['mode'], info['speechiness'], info['acousticness'], info['instrumentalness'], info['liveness'], info['valence'], info['tempo']]
			timbre = get_timbre(song_id, spotify)
			if timbre == None:
				return None
			vector.extend(timbre)
			return vector
	except Exception as e:
		print(e) 
		pass
	return None

def save_audio_feature(f):
	token = auth.get_access_token()
	spotify= spotipy.Spotify(auth=token)
	with open(f, newline='') as csvfile:
		data = list(csv.reader(csvfile))[10419:]
	print(len(data))
	count = 0
	for line in data:
		line = list(filter(None, line))
		filtered = len(line)
		if filtered < 5:
			continue
		playlist_used_id = []
		playlist_vector = []
		for song_id in line:
			if len(song_id)<5:
				continue
			v = get_vector(song_id, spotify)
			if v ==None:
				token = auth.get_access_token()
				spotify= spotipy.Spotify(auth=token)
				v = get_vector(song_id, spotify)
			if v!= None:
				playlist_vector.append(v)
				playlist_used_id.append(song_id)
		if len(playlist_used_id) > 5: 
			count = count+1
			print(count)
			with open('final_ids_used.csv', 'a+') as f:
				writer = csv.writer(f)
				writer.writerow(playlist_used_id)
			with open('final_audio_vectors.txt', 'a+') as filehandle:
				for v in playlist_vector:
					for i in v:
						filehandle.write(str(i) + ",")
					filehandle.write('\n')
				filehandle.write('\n')

file_list = 'final_thomas_perfect.csv'
save_audio_feature(file_list)

def testing():
	token = auth.get_access_token()
	spotify= spotipy.Spotify(auth=token)
	song = 'Memtrix  IC YR PAIN ' 
	t = get_track_id(song, spotify)
	print(t)


def process(f):
	with open(f, newline='') as csvfile:
		data = list(csv.reader(csvfile))
	print("processing")
	all_playlists = []
	for line in data:
		playlist = []
		perfect = False
		for i in range(len(line)):
			if line[i] == None or '':
				test = list(filter(None, line[i:]))
				if len(test) <1:
					prefect = True
				break
			else:
				playlist.append(line[i])
		if perfect != True and len(playlist)>5:
			all_playlists.append(playlist)

	count = 0
	result = []
	for line in data:
		if count > 776: 
			break
		if line == '\n':
			count = count + 1
			print(count)
		result.append(line)
	with open('final_audio_vectors.txt', 'a+') as filehandle:
		for line in result:
			for i in line:
				filehandle.write(i)
