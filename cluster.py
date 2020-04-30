
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.silhouette import silhouette
import pandas as pd
import csv
import pickle
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyClientCredentials

username = [Used Your Username]

auth = oauth2.SpotifyClientCredentials(
    client_id=[Use Your Client ID],
    client_secret = [Use Your Client Secret]
)


# Copied from sklearn.cluster.k_means_._k_init
def kpp_init(D, n_clusters, random_state_, n_local_trials=None):
    """Init n_clusters seeds with a method similar to k-means++

    Parameters
    -----------
    D : array, shape (n_samples, n_samples)
        The distance matrix we will use to select medoid indices.

    n_clusters : integer
        The number of seeds to choose

    random_state : RandomState
        The generator used to initialize the centers.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    """
    n_samples, _ = D.shape

    centers = np.empty(n_clusters, dtype=int)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state_.randint(n_samples)
    centers[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = D[centers[0], :] ** 2
    current_pot = closest_dist_sq.sum()

    # pick the remaining n_clusters-1 points
    for cluster_index in range(1, n_clusters):
        rand_vals = (
            random_state_.random_sample(n_local_trials) * current_pot
        )
        candidate_ids = np.searchsorted(
            stable_cumsum(closest_dist_sq), rand_vals
        )

        # Compute distances to center candidates
        distance_to_candidates = D[candidate_ids, :] ** 2

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(
                closest_dist_sq, distance_to_candidates[trial]
            )
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        centers[cluster_index] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def truncate(n):
	return int(n * 1000) / 1000

def weighted_distance(v1,v2):
	result = [abs(v1[0]-v2[0])*4 , abs(v1[1]-v2[1])*5 ,abs(v1[3]-v2[3])*0.3 , abs(v1[4]-v2[4])*0.2]
	result.append(abs(v1[5] - v2[5])*4)
	result.append(abs(v1[6] - v2[6])*3)
	result.append(abs(v1[7] - v2[7])*2)
	result.extend([abs(v1[9] - v2[9])*3 , abs(v1[10] - v2[10])*0.04])
	final = sum(result)
	s = 0
	for i in range(11, 23):
		s = s + abs(truncate(v1[i])-truncate(v2[i]))
	result = [0.8*final , 0.2*(s/50)]
	s = sum(result)
	return truncate(s)

def store_clusters(clusters, data, length, i):
	d = {}
	for j in range(len(clusters)):
		for index in clusters[j]:
			d[index] = j

	store_d = {}
	for j in range(len(clusters)):
		temp = []
		for index in clusters[j]:
			temp.append(data[index])
		store_d[j] = temp
	name = 'cluster_dict-' + str(i) + '.pk'
	dbfile = open(name,'wb')
	pickle.dump(store_d, dbfile)                      
	dbfile.close() 

	count = 0
	all_sequence = []
	for k in length:
		cluster_sequence = []
		for j in range(k):
			index = j+count
			cluster_sequence.append(d[index])
		all_sequence.append(cluster_sequence)
		count = count + k
	my_df = pd.DataFrame(all_sequence)
	name = 'cluster_sequences-' + str(i) + '.csv'
	my_df.to_csv(name, index=False, header=False)

def store_medoids(medoids, data, i):
	result = []
	for m in medoids:
		result.append(data[m])
	my_df = pd.DataFrame(result)
	name = 'medoids-' + str(i) + '.csv'
	my_df.to_csv(name, index=False, header=False)
	return result


with open('vectors_w_id.csv', newline='') as csvfile:
	raw_data = list(csv.reader(csvfile))[:10000]
data = []
length = []
c = 0
for line in raw_data:
	l = list(filter(None, line))
	if len(l)>0:
		temp = []
		for i in range(len(l)-1):
			temp.append(float(l[i]))
		data.append(temp)
		c = c+1
	else:
		length.append(c)
		c = 0

l=len(data)
#print(data)

X = []
for i in range(l):
	X.append(np.zeros(l))

for i in range(l):
	#print(i)
	for j in range(l):
		if i ==j:
			X[i][j] = 0
		elif X[j][i] != 0:
			X[i][j] = X[j][i]
		else:
			d = weighted_distance(data[i], data[j])
			X[i][j] = np.float16(d)

metric = distance_metric(type_metric.USER_DEFINED, func=weighted_distance);
random_state = check_random_state(None)
print('success')

options = [30]
for i in range(len(options)):
	print("enter round " + str(options[i]))
	initial_medoids = kpp_init(np.array(X), options[i], random_state)
	print(initial_medoids)
	kmedoids_instance = kmedoids(X, initial_medoids, data_type='distance_matrix');
	kmedoids_instance.process()
	clusters = kmedoids_instance.get_clusters()
	store_clusters(clusters, data, length, options[i])
	medoids = kmedoids_instance.get_medoids()
	medoids_vectors = store_medoids(medoids, data, options[i])
	score = silhouette(data, clusters, metric = metric).process().get_score()
	print(sum(score)/len(score))

	visual= cluster_visualizer_multidim()
	visual.append_clusters(clusters, data)
	visual.show(pair_filter=[[0, 10], [1, 10], [3,10], [4,10], [5,10],[6,10], [7,10], [9,10], [0,1]])
	visual.show()

def append_id_on_vector():
	with open('clean_ids.csv', newline='') as csvfile:
		id_d = list(csv.reader(csvfile))
	id_data = []
	length = []
	for line in id_d:
		playlist = list(filter(None, line))
		if len(playlist)>0:
			id_data.append(playlist)
			length.append(len(playlist))

	f = open('final_audio_vectors.txt','r')
	raw_data = f.readlines()

	data = []
	count = 0
	playlist_divided = []
	i = 0
	j = 0
	visited = False
	for line in raw_data:
		if len(line) <5:
			if visited == False and i == 145:
				visited = True
			else:
				data.append([])
				playlist_divided.append(j)
				i = i+1
			j = 0
			continue
		if i == 145 and visited == False:
			continue
		v = line.split(',')[:-1]
		vector = []
		for num in v: 
			if 'e' in num:
				vector.append(0)
			else:
				vector.append(float(num))
		vector.append(id_data[i][j])
		data.append(vector)
		j = j+1

	my_df = pd.DataFrame(data)
	my_df.to_csv('vectors_w_id.csv', index=False, header=False)


#All previous attempts on k-means clustering and t-SNE visualization. 
#NOT USED FOR FINAL PRODUCT. 
def kmeans_cluster():
	with open('id.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	pool = []
	playlist_length = []
	for line in data:
		length = 0
		for song_id in line:
			if song_id != '':
				try: 
					info = spotify.audio_features(song_id)[0]
					if info!= None:
						vector = [info['danceability'], info['energy'], info['key'], info['loudness'], info['mode'], info['speechiness'], info['acousticness'], info['instrumentalness'], info['liveness'], info['valence'], info['tempo']]
						pool.append(vector)
						length = length + 1
				except Exception as e: 
					print(e)
					token = auth.get_access_token()
					spotify= spotipy.Spotify(auth=token)
					pass
		playlist_length.append(length)

	print("entering kmeans")

	kmeans = KMeans(n_clusters=20).fit(pool)
	i = 0
	count = 0
	lines = []
	for l in playlist_length:
		line = []
		while  i < l + count:
			line.append(kmeans.labels_[i])
			i = i+1
		count = count + l
		lines.append(line)
	cluster = pd.DataFrame(lines)
	cluster.to_csv('cluster_index1.csv', index=False, header=False)

	with open('cluster_center1.csv','a+', newline ='') as f:
		writer = csv.writer(f)
		writer.writerow(kmeans.cluster_centers_)

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def extract(lst, i): 
    return np.array([item[i] for item in lst])

def visualize():
	with open('id.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	pool = []
	count = 0
	for line in data:
		for song_id in line:
			if song_id != '':
				try: 
					info = spotify.audio_features(song_id)[0]
					if info!= None:
						vector = [info['danceability'], info['energy'], info['key'], info['loudness'], info['mode'], info['speechiness'], info['acousticness'], info['instrumentalness'], info['liveness'], info['valence'], info['tempo']]
						pool.append(vector)
						count = count + 1
						if count >= 5000:
							break
				except Exception as e: 
					print("failed")
					pass
	with open('cluster_index1.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	assign = []
	count = 0
	for line in data:
		for item in line:
			if item !='':
				assign.append(float(item))
				count = count + 1
				if count >= 5000:
					break

	with open('cluster_center_new.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	
	X_embedded = TSNE(n_components=2).fit_transform(data)
	x = extract(X_embedded,0)
	y = extract(X_embedded,1)
	plt.scatter(x, y)
	#plt.colorbar(ticks=range(20))
	plt.clim(-0.5, 9.5)
	plt.show()

def avg_v(l):
	result = np.zeros(11)
	for v in l:
		for i in range(0,len(v)):
			result[i] = result[i] + v[i]
	length = len(result)
	for i in range(0, length):
		result[i] = result[i]/length
	return result

def normalize(l):
	rearrange = []
	for i in range(11):
		rearrange.append(extract(l, i))
	for item in rearrange:
		std = statistics.stdev(item)
		if std == 0:
			std = 1
		avg = sum(item)/len(item)
		for num in item:
			num = (num-avg)/std
	result = []
	for k in range(0, len(l)):
		vector = []
		for i in range(11):
			vector.append(rearrange[i][k])
		result.append(vector)
	return result


def test():
	with open('id.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	pool = []
	avg_list = []
	count = 0
	playlist_length = []
	for line in data:
		playlist = []
		for song_id in line:
			if song_id != '':
				try: 
					info = spotify.audio_features(song_id)[0]
					if info!= None:
						vector = [info['danceability'], info['energy'], info['key'], info['loudness'], info['mode'], info['speechiness'], info['acousticness'], info['instrumentalness'], info['liveness'], info['valence'], info['tempo']]
						playlist.append(vector)
						count = count + 1
						if count >=1000:
							break
				except Exception as e: 
					print("failed")
					pass
		playlist_length.append(len(playlist))
		if count >= 1000:
			break
		if len(playlist)>1:
			#result = normalize(playlist)
			#pool.extend(result)
			pool.extend(playlist)

	X_embedded = TSNE(n_components=2).fit_transform(pool)
	x = extract(X_embedded,0)
	y = extract(X_embedded,1)
	colors = assign[:len(x)]
	print(len(x))
	plt.scatter(x, y, c = colors)
	#plt.colorbar(ticks=range(20))
	plt.clim(-0.5, 9.5)
	plt.show()



def convert_dict():
	with open('cluster_index_normalize.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	d = {}
	d['cluster'] = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
	key = []
	value = []
	pos = []
	for line in data:
		line = list(filter(None, line))
		i = 0
		while i+4<len(line):
			substring = line[i:i+3]
			r = []
			for s in substring:
				r.append(int(float(s)))
			position = (i+4)/len(line)
			pos.append(position)
			key.append(np.array(r))
			value.append(int(float(line[i+4])))
			i = i+1

	d['train_input'] = np.array(key)
	d['train_target'] = np.array(value)
	d['position'] = np.array(pos)
	dbfile = open('model_data_normalize.pk','wb')
	pickle.dump(d, dbfile)                      
	dbfile.close() 

def cluster():
	with open('id.csv', newline='') as csvfile:
		data = list(csv.reader(csvfile))
	pool = []
	playlist_length = []
	used_id = []
	for line in data:
		playlist = []
		playlist_id = []
		for song_id in line:
			if song_id != '':
				try: 
					info = spotify.audio_features(song_id)[0]
					if info!= None:
						vector = [info['danceability'], info['energy'], info['key'], info['loudness'], info['mode'], info['speechiness'], info['acousticness'], info['instrumentalness'], info['liveness'], info['valence'], info['tempo']]
						playlist.append(vector)
						playlist_id.append(song_id)
				except Exception as e: 
					print(e)
					token = auth.get_access_token()
					spotify= spotipy.Spotify(auth=token)
					pass
		if len(playlist) >1:
			playlist_length.append(len(playlist))
			used_id.append(playlist_id)
			result = normalize(playlist)
			pool.extend(result)

	i = pd.DataFrame(used_id)
	i.to_csv('cluster_used_id.csv', index=False, header=False)

	print("entering kmeans")

	kmeans = KMeans(n_clusters=50).fit(pool)
	i = 0
	count = 0
	lines = []
	for l in playlist_length:
		line = []
		while  i < l + count:
			line.append(kmeans.labels_[i])
			i = i+1
		count = count + l
		lines.append(line)
	cluster = pd.DataFrame(lines)
	cluster.to_csv('cluster_index_normalize-50.csv', index=False, header=False)

	try: 
		cluster = pd.DataFrame(kmeans.cluster_centers_)
		cluster.to_csv('cluster_center_normalize-50.csv', index=False, header=False)
	except:
		print("attempt failed")
		with open('cluster_center_normalize-50.csv','a+', newline ='') as f:
			writer = csv.writer(f)
			writer.writerow(kmeans.cluster_centers_)

	X_embedded = TSNE(n_components=2).fit_transform(pool)
	x = extract(X_embedded,0)
	y = extract(X_embedded,1)
	colors = kmeans.labels_
	print(len(x))
	plt.scatter(x, y, c = colors)
	#plt.colorbar(ticks=range(20))
	plt.clim(-0.5, 9.5)
	plt.show()
