import numpy as np
import pandas as pd
import keras
import random
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
import csv
import pickle
import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import os
from bottle import route, run, request

MIN = 0

username = [Use your username]

auth = oauth2.SpotifyClientCredentials(
    client_id=[Use your Client ID],
    client_secret = [Use your Client Secret]
)

def get_vector(song_id, spotify):
	try: 
		info = spotify.audio_features(song_id)[0]
		if info!= None:
			vector = [info['danceability'], info['energy'], info['key'], info['loudness'], info['mode'], info['speechiness'], info['acousticness'], info['instrumentalness'], info['liveness'], info['valence'], info['tempo']]
			return vector
	except Exception as e:
		print(e) 
		pass
	return None

def convert_id(ids):
	token = auth.get_access_token()
	spotify= spotipy.Spotify(auth=token)
	songs = []
	for i in ids:
		track = spotify.track(i)
		songs.append(track['name'] + ' artist:' + track['artists'][0]['name'])
		v = get_vector(i,spotify)
		print(v)
	return songs

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.step = None
        self.sorted = False
        self.optimal_previous = None
        self.optimal_distance = None
        self.cluster_index = None
        self.cluster_weight = None
        self.visited = False

def convert_data(data):
	n_gram = []
	for line in data:
		line = list(filter(None, line))
		i = 0
		while i+4 <= len(line):
			substring = line[i:i+3]
			position = (i+3)/len(line)
			substring.append(position)
			substring.append(line[i+3])
			i = i+1
		n_gram.append(substring)
	return n_gram 

def language_model(X_tr, y_tr, X_val, y_val, size):
	model = Sequential()
	model.add(Embedding(size, 50, input_length=4, trainable=True))
	model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
	model.add(Dense(size, activation='softmax'))
	#print(model.summary())
	model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
	model.fit(X_tr, y_tr, epochs=100, verbose=2, validation_data=(X_val, y_val))
	return model 

def generate_tree(model, cluster_size, length):
	seed_cluster = []
	for i in range(3):
		number = random.randint(0, cluster_size-1)
		seed_cluster.append(number)
	root = Tree()
	root.cluster_index = seed_cluster[-1]
	root.step = 3/length
	i = 1
	#print(seed_cluster)
	recurse_tree(model, root, seed_cluster, i, length)
	return root 

def print_tree(root):
	print(root.cluster_index)
	if root.left != None:
		print_tree(root.left)
	if root.right!=None:
		print_tree(root.right)
	return None 

def recurse_tree(model, root, previous, i, length):
	if i == length:
		return None
	previous.append(i/length)
	yhat = model.predict(np.array([previous]), batch_size=32, verbose=0)[0]
	#np.put(yhat, first_index, 0)
	first = max(yhat)
	first_index = np.argmax(yhat)
	np.put(yhat, first_index, 0)
	second = max(yhat)
	second_index = np.argmax(yhat)
	both = first + second
	
	root.left = Tree()
	root.left.parent = root
	root.left.cluster_index = first_index
	root.left.cluster_weight = (both-first)/both
	root.left.step = i/length
	left_previous = previous[1:-1]
	left_previous.append(first_index)
	recurse_tree(model, root.left, left_previous, i+1, length)
	
	root.right = Tree()
	root.right.parent = root
	root.right.cluster_index = second_index
	root.right.cluster_weight = (both-second)/both 
	root.right.step = i/length
	right_previous = previous[1:-1]
	right_previous.append(second_index)
	recurse_tree(model, root.right, right_previous, i+1, length)
	return None


def weighted_distance(v1,v2):
	result = [abs(v1[0]-v2[0])*4 , abs(v1[1]-v2[1])*5 ,abs(v1[3]-v2[3])*0.3 , abs(v1[4]-v2[4])*0.2]
	result.append(abs(v1[5] - v2[5])*4)
	result.append(abs(v1[6] - v2[6])*3)
	result.append(abs(v1[7] - v2[7])*2)
	result.extend([abs(v1[9] - v2[9])*3 , abs(v1[10] - v2[10])*0.04])
	final = sum(result)
	s = 0
	for i in range(11, 23):
		s = s + abs(v1[i]-v2[i])
	final = [0.8*final , 0.2*(s/50)]
	s = sum(final)
	return s

def prepare_data(file, cluster_number):
	with open(file) as csvfile:
		data = list(csv.reader(csvfile))
	n_gram = np.array(convert_data(data))
	X, y = n_gram[:,:-1], n_gram[:,-1]
	y = [int(float(t)) for t in y]
	y = to_categorical(y, num_classes=cluster_number)
	return train_test_split(X, y, test_size = 0.1)

def save_model(cluster_size, playlist_len, file):
	X_tr, X_val, y_tr, y_val = prepare_data(file, cluster_size)
	model = language_model(X_tr, y_tr, X_val, y_val, cluster_size)
	model.save('model.h5')

def check_criteria(v):
	#return True
	if float(v[0]) < 0.6:
		return False
	if float(v[10]) < 80:
		return False
	if float(v[5]) > 0.25:
		return False
	if float(v[9]) <0.3:
		return False
	if float(v[1]) <0.4:
		return False
	return True

def select_pool(number):
	with open('vectors_w_id.csv') as csvfile:
		data = list(csv.reader(csvfile))[50000:]
	selected = []
	i = 0
	while i < number:
		index = random.randint(0, len(data)-1)
		while list(filter(None,data[index])) == []:
			index = random.randint(0, len(data)-1)
		v = data[index]
		if check_criteria(v):
			selected.append(v)
			i = i+1
	return selected

def assign_clusters(vectors, medoid_file, dict_file):
	with open(medoid_file, newline='') as csvfile:
		data = list(csv.reader(csvfile))
	data = [x for x in data if list(filter(None,x))!= []]
	clusters = []
	for i in range(len(data)):
		clusters.append([])
	for v in vectors:
		minimum = -1
		min_i = 0
		for i in range(len(data)):
			d = weighted_distance(convert_number(v[:23]), convert_number(data[i][:23]))
			if minimum == -1 or d<minimum:
				minimum = d
				min_i = i 
		clusters[min_i].append(v)
	
	#make sure all clusters have at least one 
	with open('id_dict.pk', 'rb') as h:
		id_d = pickle.load(h)
	
	with open(dict_file, 'rb') as handle:
		d = pickle.load(handle)
	for i in range(len(clusters)):
		if len(clusters[i]) == 0:
			index = random.randint(0,len(d[i]))
			v= d[i][index]
			s=  0
			for k in range(10):
				s = s+ v[k]
			v.append(id_d[s])
			clusters[i].append(v)

	return clusters

def generate_dict():
	with open('vectors_w_id.csv') as f:
		data = list(csv.reader(f))
	d = {}
	for line in data:
		line = list(filter(None, line))
		if line!=[]:
			s = 0
			for i in range(10):
				s = s+float(line[i])
			d[s] = line[-1]
	name = 'id_dict.pk'
	dbfile = open(name,'wb')
	pickle.dump(d, dbfile)                      
	dbfile.close() 


def convert_number(line):
	new = []
	for i in range(len(line)):
		v = line[i]
		if isinstance(v, str):
			if 'e' in v:
				new.append(0)
			else:
				new.append(float(v))
		else:
			new.append(v)
	return new


def get_transition_weight(v1, v2, step):
	#if two songs are the same, return large weight
	if v1[-1] == v2[-1]:
		return 1000
	
	v1 = convert_number(v1[:23])
	v2 = convert_number(v2[:23])

	#preferred key change has weight 0 
	w2 = 4
	if v1[4] == v2[4]:
		sub = (v1[2] + 5)%12 
		dom = (v1[2] + 7)%12
		if v2[2] == sub or dom:
			w2 = 0
		if v2[4] == v1[4]:
			w2 = 2
	else:
		if v1[2] == v2[2]:
			w2 = 0
		elif v1[4] == 0 and v2[2] == (v1[2]-3)%12:
			w2 = 0
		elif v2[2] == (v1[2]+3)%12:
			w2 = 0

	#preferred variant speechiness
	w5 = 1
	if v1[5] > 0.09:
		if v2[5]< 0.06:
			w5 = 0.1
	elif v1[5] < 0.04:
		if v2[5]-v1[5]>0.02:
			w5 = 0.1
	else:
		check = abs(v2[5]-v1[5])
		if check < 0.05:
			w5 = check*10

	w9 = 1
	if v1[9] > 0.8:
		if v2[9]<0.5:
			w9 =0.1
	elif v2[9] < 0.2: 
		if v2[9]>0.5:
			w9 = 0.1
	else: 
		w9 = 0.5

	w7 = 2
	if v1[7]>0.8:
		if v2[7]<0.1:
			w7 = 0.1
	elif v1[7] <0.2:
		if v2[7]>0.6:
			w7 = 0.1
	else:
		if abs(v2[7]-v1[7]) < 0.1:
			w7 = 1
		else:
			w7 = abs(v2[7]-v1[7])

	w6 = 1
	if v1[6]<0.1:
		if v2[6]>0.5:
			w6 = 0.1
		elif v2[6] > v1[6]+0.2:
			w6 = 0.2
	elif v1[6]>0.7:
		if v2[6]< 0.1:
			w6 = 0.1
		elif v2[6] < v1[6]-0.2:
			w6 = 0.2
	else:
		if abs(v2[7]-v1[7]) < 0.05:
			w7 = 0.5
		else:
			w6 = abs(v2[6]-v1[6])

	w1 = 2
	w3 = 1
	w10 = 5
	if step < 0.85:

		if v1[1] > 0.7:
			if abs(v1[1]-v2[1]) <0.2:
				w1 = 0.2
		elif v2[1] - v1[1] > 0.1:
			w1 = 0.1
		elif v2[1] - v1[1]  > 0:
			w1 = 0.2

		if v1[3] > -5:
			if abs(v1[1]-v2[1]) <3:
				w3 = 0.2
		if v2[3] - v1[3] >1:
			w3 = 0.1
		elif v2[3] - v1[3] >0:
			w3 = 0.2
		
		if v1[10] > 130:
			if abs(v1[10]-v2[10]) <10:
				w10 = 1
		if v2[10]-v1[10] > 5 and v2[10]-v1[10] < 30:
			w10 = 0.1
		elif v2[10]-v1[10] > 30:
			w10 = 1

	else:

		if v1[1] < 0.5:
			if abs(v1[1]-v2[1]) <0.2:
				w1 = 0.2
		elif v1[1] - v2[1] > 0.1:
			w1 = 0.1
		elif v1[1] - v2[1]  > 0:
			w1 = 0.2

		if v1[3] < -10:
			if abs(v1[1]-v2[1]) <3:
				w3 = 0.2
		if v1[3] - v2[3] > 0.1:
			w3 = 0.1
		elif v1[3] - v2[3] >0:
			w3 = 0.2
		
		if v1[10] < 100:
			if abs(v1[10]-v2[10]) <10:
				w10 = 1
		if v1[10]-v2[10] > 10 and v1[10]-v2[10] < 30:
			w10 = 0.1
		elif v1[10]-v2[10] > 30:
			w10 = 1
	
	weight = w2 + w5 + w9 + w7 + w6 + w1 + w3 + w10
	s = 0
	for i in range(11, 23):
		s = s + abs(v1[i]-v2[i])
	final = weight*0.6 + (s/50)*0.4
	return final

def find_optimal(optimal_distance, C1, C2, cluster_weight, step):
	result_vector = []
	best_vector = []
	for j in range(len(C2)):
		result = -1
		index = 0
		for i in range(len(C1)):
			w = cluster_weight * get_transition_weight(C1[i], C2[j], step)
			if optimal_distance == None:
				v = 0
				v_i1 = 0
			else:
				v = optimal_distance[i]
				if i+1<len(C1):
					v_i1 = optimal_distance[i+1]
				else:
					v_i1 = 0
			if result == -1 or w+v < result:
				result = w+v
				best = C1[i]
			if result <= v_i1 + MIN:
				break
		result_vector.append(result)
		best_vector.append(best)
	return [result_vector, best_vector]	


def traverse_tree(root, clusters, left):
	if root.left != None:
		if left == True: 
			next_node = root.left
		else: 
			next_node = root.right
		if next_node.visited == True:
			#root.visited = True
			return None,None,None
		#print('entering' + str(root.cluster_index)+ ' , ' + str(root.step))
		C1 = clusters[root.cluster_index]
		#print(next_node.cluster_index)
		C2 = clusters[next_node.cluster_index]
		if root.sorted == False and root.optimal_distance != None:
			zipped = sorted(zip(root.optimal_distance, root.optimal_previous, C1))
			root.optimal_distance = [x for x,_,_ in zipped]
			root.optimal_previous = [x for _,x,_ in zipped]
			C1 = [x for _,_,x in zipped]
			clusters[root.cluster_index] = C1
			root.sorted = True
		result = find_optimal(root.optimal_distance, C1, C2, next_node.cluster_weight, next_node.step)
		next_node.optimal_distance = result[0]
		next_node.optimal_previous = result[1]
		d1,p1,s1 = traverse_tree(next_node, clusters, True)
		d2,p2,s2 = traverse_tree(root, clusters, False)
	else:
		root.visited = True
		d = min(root.optimal_distance)
		i = root.optimal_distance.index(d)
		p = root.optimal_previous[i]
		s = [clusters[root.cluster_index][i][-1]]
		return d,p,s
	
	root.visited = True
	if d2 == None:
		return d1,p1,s1
	if d1<=d2:
		i = clusters[root.cluster_index].index(p1)
		s1.append(p1[-1])
		if root.optimal_previous ==None:
			return s1
		p1 = root.optimal_previous[i]
		#print(d1,p1,s1)
		return d1,p1,s1
	else:
		i = clusters[root.cluster_index].index(p2)
		s2.append(p2[-1])
		if root.optimal_previous ==None:
			return s2
		p2 = root.optimal_previous[i]
		#print(d2,p2,s2)
		return d2,p2,s2

def main():
	cluster_size = 30
	playlist_len = 10
	song_pool = 1000
	dict_file = 'cluster_dict-30.pk'
	file = 'cluster_sequences-30.csv'
	#save_model(cluster_size, playlist_len, file)
	vectors = select_pool(song_pool)
	print("assigning clusters")
	clusters = assign_clusters(vectors, 'medoids-30.csv', dict_file)
	print("generating tree")
	model = load_model('model.h5')
	tree = generate_tree(model, cluster_size, playlist_len)
	print('enter traversing')
	playlist_id = traverse_tree(tree, clusters, True)
	songs = convert_id(list(reversed(playlist_id)))
	print(songs)

main()

# Generating random sequence
def rand():
	for i in range(10):
		t = random.randint(1,5)
		print(t)

