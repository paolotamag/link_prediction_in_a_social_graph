#Paolo Tamagnini - 1536242
import csv
import time
import pprint as pp
import networkx as nx
import itertools as it
import numpy as np

print
print "Current time: " + str(time.asctime(time.localtime()))
print


def create_graph(input_graph_path):
	g = nx.Graph()
	input_file = open(input_graph_path, 'r')
	input_file_csv_reader = csv.reader(input_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_NONE)
	for line in input_file_csv_reader:
		if len(line) != 3:
			continue
		g.add_edge(*[int(line[0]), int(line[1])])
	input_file.close()
	return g


def create_test_graph(input_test_graph_path, g_training):
	g_test = create_graph(input_test_graph_path)
	all_nodes_in_test_graph = g_test.nodes()
	for node_in_test in all_nodes_in_test_graph:
		if node_in_test not in g_training:
			g_test.remove_node(node_in_test)
	return g_test





def create_bunch_of_edges_of_interest(g_training):
	all_test_edges = []
	all_nodes_in_g_training = g_training.nodes()
	all_nodes_in_g_training.sort()
	for u_v in it.combinations(all_nodes_in_g_training, 2):
		if u_v[1] in g_training[u_v[0]]:
			continue
		all_test_edges.append(u_v)
	return all_test_edges
	



def sum_values_in_dictionaries(dic_sum, dic):
	for k in dic:
		if k not in dic_sum:
			dic_sum[k] = 0.
		dic_sum[k] += dic[k]
	return

def divide_values_in_dictionaries(dic, divisor=1):
	for k in dic:
		dic[k] /= divisor
	return



def write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor):
	output_file = open(output_file_name, 'w')
	output_file_csv_writer = csv.writer(output_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
	output_file_csv_writer.writerow(['k%', 'P@k_improvement_wrt_Random_Predictor'])
	all_keys = avg_precision_improvement_wrt_random_predictor.keys()
	all_keys.sort()
	for key in all_keys:
		output_file_csv_writer.writerow([key, avg_precision_improvement_wrt_random_predictor[key]])
	output_file.close()
	return












# score(u,v) := Jaccard(u, v)
def link_prediction_with_jaccard(g_training, bunch_of_edges_of_interest):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		U = set(g_training.neighbors(u))
		V =  set(g_training.neighbors(v))
		intersc = len(U & V)
		union = len(U | V)
		score = intersc/float(union)
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)
	return result



# score(u,v) := Adamic_Adar(u, v)
def link_prediction_with_adamic_adar(g_training, bunch_of_edges_of_interest):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		U = set(g_training.neighbors(u))
		V =  set(g_training.neighbors(v))
		interset = list(U & V)
		score = 0
		for i in interset:
			score = score + 1/np.log(len(g_training.neighbors(i)))
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)
	return result
	


# score(u,v) := Preferential_Attachment(u, v)
def link_prediction_with_preferential_attachment(g_training, bunch_of_edges_of_interest):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		score = len(g_training.neighbors(u)) * len(g_training.neighbors(v))
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)
	return result



# score(u,v) := Common neighbors between u and v
def link_prediction_with_number_of_common_neighbors(g_training, bunch_of_edges_of_interest):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		U = set(g_training.neighbors(u))
		V =  set(g_training.neighbors(v))
		score = len(U & V)
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print	

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)
	return result



# score(u,v) := -(length of shortest path from u to v)
def link_prediction_with_shortest_path(g_training, bunch_of_edges_of_interest):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	numbOfTotNodes = len(g_training.nodes())
	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		try:
			score = -len(nx.shortest_path(g_training,source=u,target=v))
		except nx.NetworkXNoPath:
			score = -numbOfTotNodes
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)
	return result



### Perfect-Predictor!!!
def link_prediction_with_PERFECT_PREDICTOR(g_training, bunch_of_edges_of_interest, g_test):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	edgesTest = g_test.edges()
	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		if (u,v) in edgesTest:
			score = 1.0
		else:
			score = 0.0
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)

	return result



### Worst-Predictor :\
def link_prediction_with_WORST_PREDICTOR(g_training, bunch_of_edges_of_interest, g_test):
	result = []

	# 'result' is a list of lists, or tuples, of three element: [u, v, score].
	# 'result' must contain exactly the same edges contained in 'bunch_of_edges_of_interest'.

	edgesTest = g_test.edges()
	h = time.time()
	c = 0
	maximus = len(bunch_of_edges_of_interest)
	print
	for u,v in bunch_of_edges_of_interest:
		line_in_result = []
		if (u,v) in edgesTest:
			score = 0.0
		else:
			score = 1.0
		line_in_result.append(u)
		line_in_result.append(v)
		line_in_result.append(score)
		result.append(line_in_result)
		c = c + 1
		if time.time() - h > 5 or c == maximus:
			h = time.time()
			g = float(c)/maximus*100
			print '[','','~'*int(g/10),' '*int(10-g/10),']',int(g), '%'
	print

	# Sort the list in descending order of prediction.
	# A simple and NOT INFORMATIVE tie-breaking rule is applied.
	result.sort(key=lambda x: (x[2], x[0], -x[1]), reverse=True)
	return result









def precision_improvement_wrt_random_predictor(predicted_edges_rank, g_test):
	improved_precision_at_k_wrt_random_predictor = {}

	# 'improved_precision_at_k_wrt_random_predictor' is a python dictionary in which:
	# 1) Keys are INTEGERS in [1, 100] representing 1%, 2%, 3%, ..., 100% of 'predicted_edges_rank'.
	# 2) Values are the corresponding improvements of P@k% with respect to a random predictor.
	# 
	# Example:
	#   improved_precision_at_k_wrt_random_predictor[3] = P@3%(A_Particular_Link_Prediction_method) / P@3%(Random_Predictor)

	# !RULE: Calculate P@k%(Random_Predictor) without creating a Random predictor!


	cT = float(len(g_test.edges()))
	nB = len(predicted_edges_rank)
	nBComp = int(nB / 100.0)
	pofSuccess = cT/nB
	testEdgeYetToBeFound = set(g_test.edges())
	cOld = 0
	intrsctOld = set([])

	for kr in xrange(1,101):
		
		k = kr * nBComp		
		kOld = (kr-1) * nBComp		
		newEdgesToInspect = predicted_edges_rank[kOld:k]		
		newEdgesToInspect = set([(ch[0],ch[1]) for ch in newEdgesToInspect])		
		testEdgeYetToBeFound = testEdgeYetToBeFound - intrsctOld		
		intrsct = newEdgesToInspect & testEdgeYetToBeFound		
		c = len(intrsct) + cOld			   
		cOld = c		
		intrsctOld = intrsct		
		improved_precision_at_k_wrt_random_predictor[kr] = (c / float(k)) / pofSuccess

	return improved_precision_at_k_wrt_random_predictor



















################################################################################################################
################################################################################################################
################################################################################################################

#graph_name = "BUP" # Also useful for debug ;)
#graph_name = "UAL"
#graph_name = "INF"
#graph_name = "SMG"
#graph_name = "EML"
graph_name = "YST"
#graph_name = "FBK"


# Dictionaries that will contain the final results.
avg_precision_improvement_wrt_random_predictor_using_jaccard = {}
avg_precision_improvement_wrt_random_predictor_using_adamic_addar = {}
avg_precision_improvement_wrt_random_predictor_using_preferential_attachment = {}
avg_precision_improvement_wrt_random_predictor_using_number_of_common_neighbors = {}
avg_precision_improvement_wrt_random_predictor_using_shortest_path = {}
avg_precision_improvement_wrt_random_predictor_using_PERFECT_PREDICTOR = {}
avg_precision_improvement_wrt_random_predictor_using_WORST_PREDICTOR = {}



for index in xrange(5):
	training_set_graph_name = './networks/'+graph_name+'_train_'+str(index)+'.net'
	test_set_graph_name     = './networks/'+graph_name+'_test_'+str(index)+'.net'
	
	print
	print
	print "-----------------------------------------------"
	print " Current data: "
	print "               " + training_set_graph_name
	print "               " + test_set_graph_name
	print
	
	pp.pprint("Load Training Graph.")
	print
	g_training = create_graph(training_set_graph_name)
	print " #Nodes in Training Graph= " + str(len(g_training))
	print " #Edges in Training Graph= " + str(g_training.number_of_edges())
	print
	print
	pp.pprint("Load Test Graph.")
	print
	g_test = create_test_graph(test_set_graph_name, g_training)
	print " #Nodes in Test Graph= " + str(len(g_test))
	print " #Edges in Test Graph= " + str(g_test.number_of_edges())
	print
	
	
	# create the bunch of edges of interest
	print
	print
	pp.pprint("Creation of the bunch of edges of interest.")
	bunch_of_edges_of_interest = create_bunch_of_edges_of_interest(g_training)
	
	
	
	# Start Predictions!
	
	current_method = 'jaccard'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_jaccard = link_prediction_with_jaccard(g_training, bunch_of_edges_of_interest)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	current_method = 'adamic_adar'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_adamic_adar = link_prediction_with_adamic_adar(g_training, bunch_of_edges_of_interest)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	current_method = 'preferential_attachment'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_preferential_attachment = link_prediction_with_preferential_attachment(g_training, bunch_of_edges_of_interest)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	current_method = '#Common_neighbours'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_number_of_common_neighbors = link_prediction_with_number_of_common_neighbors(g_training, bunch_of_edges_of_interest)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	current_method = '-(shortest_path)'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_shortest_path = link_prediction_with_shortest_path(g_training, bunch_of_edges_of_interest)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	current_method = 'PERFECT_PREDICTOR'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_perfect_predictor = link_prediction_with_PERFECT_PREDICTOR(g_training, bunch_of_edges_of_interest, g_test)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	current_method = 'WORST_PREDICTOR'
	print "Current time: " + str(time.asctime(time.localtime()))
	pp.pprint("Link Prediction for the bunch of edges of interest using '"+current_method+"'.")
	result_from_worst_predictor = link_prediction_with_WORST_PREDICTOR(g_training, bunch_of_edges_of_interest, g_test)
	print "Current time: " + str(time.asctime(time.localtime()))
	print
	
	
	
	
	
	
	
	# Sum and store all P@k improvements for all techniques.
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_jaccard, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_jaccard, improved_p_at_k)
	
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_adamic_adar, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_adamic_addar, improved_p_at_k)
	
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_preferential_attachment, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_preferential_attachment, improved_p_at_k)
	
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_number_of_common_neighbors, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_number_of_common_neighbors, improved_p_at_k)
	
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_shortest_path, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_shortest_path, improved_p_at_k)
	
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_perfect_predictor, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_PERFECT_PREDICTOR, improved_p_at_k)
	
	improved_p_at_k = precision_improvement_wrt_random_predictor(result_from_worst_predictor, g_test)
	sum_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_WORST_PREDICTOR, improved_p_at_k)
	
	



# Compute AVG P@k improvements for all techniques.
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_jaccard, divisor=5)
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_adamic_addar, divisor=5)
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_preferential_attachment, divisor=5)
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_number_of_common_neighbors, divisor=5)
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_shortest_path, divisor=5)
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_PERFECT_PREDICTOR, divisor=5)
divide_values_in_dictionaries(avg_precision_improvement_wrt_random_predictor_using_WORST_PREDICTOR, divisor=5)




# Write on file all AVG P@k improvements for all techniques.

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"jaccard"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_jaccard)

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"adamic_addar"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_adamic_addar)

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"preferential_attachment"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_preferential_attachment)

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"number_of_common_neighbors"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_number_of_common_neighbors)

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"shortest_path"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_shortest_path)

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"PERFECT_PREDICTOR"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_PERFECT_PREDICTOR)

output_file_name = "./results/"+graph_name+"__P_at_k__improvement__method_"+"WORST_PREDICTOR"+".csv"
write_avg_precision_improvement_wrt_random_predictor_on_file(output_file_name, avg_precision_improvement_wrt_random_predictor_using_WORST_PREDICTOR)





print
print
print "Current time: " + str(time.asctime(time.localtime()))
print " Done ;)"
print













