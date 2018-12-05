import time
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator



# Read input with the format given for the project
def read_input(filename):
    f = open(filename)
    
    l = []
    
    for line in f:
        l.append(line)
        
    nb_tasks = int(l[0])
    l = l[1:]
    new_l = []
    
    for elem in l:
        new_l.append([int(x) for x in elem.split(' ')])
    
    
    
    return (nb_tasks, dict(zip(['A', 'B', 'C'], new_l)))

# Perform necessary compuations in order to update the 'dates' of the permutation or sub-permutation
def update_dates(t_pi, d_k):
    
    t_pi_a, t_pi_b, t_pi_c = t_pi
    d_k_a, d_k_b, d_k_c = d_k

    delta_1 = max(0, d_k_a - (t_pi_b - t_pi_a))
    delta_2 = max(0, (d_k_b + delta_1) - (t_pi_c - t_pi_b))
    
    return ((t_pi_a + d_k_a), (t_pi_b + d_k_b + delta_1), (t_pi_c + d_k_c + delta_2))

def get_ord_end(ordo, tasks_lengths):
    dates = (0, 0, 0)
    
    for t in ordo:
        d_k = (tasks_lengths['A'][t], tasks_lengths['B'][t], tasks_lengths['C'][t])
        dates = update_dates(dates, d_k)
    
    *d, res = dates
    
    return res

#############################################################################################
#                                                                                           #
#                              APPROXIMATED JOHNSON IMPLEMENTATION                          #
#                                                                                           #
#############################################################################################

# A very simple implementation of the Johson algorithm. Exact for instances of the problem with
# two tasks, 2-approximate for 3 tasks.
def johnson(nb_tasks, tasks_durations):
    res_g = []
    res_d = []
    tasks_list = list(range(nb_tasks))
    # lol à corriger
    while len(tasks_list) > 0:
        x_A = None
        x_B = None
        for t in tasks_list:
            if x_A is None or tasks_durations['A'][t] < tasks_durations['A'][x_A]:
                x_A = t
            if x_B is None or tasks_durations['B'][t] < tasks_durations['B'][x_B]:
                x_B = t
        
        if x_A < x_B:
            task_index = x_A
            res_g = res_g + [task_index]
            tasks_list.remove(task_index)
        else:
            task_index = x_B
            res_d = [task_index] + res_d
            tasks_list.remove(task_index)
            
    res = res_g + res_d
    return (get_ord_end(res, tasks_durations), res)

#############################################################################################
#                                                                                           #
#                                     FIRST TREE IMPLEMENTATION                             #
#                                                                                           #
#############################################################################################

# This first implementation is extremly naïve and was used only as a proof of concept
# It first creates the whole tree and compute every lower bound, taking quite a lot of
# memory and being very slow for big instances.

# A very simple class representing a Node.
# Used by the Tree class
class Node:
    def __init__(self, label, val, inf, sup):
        self.label = label
        self.val = val
        self.inf = inf
        self.children = []

    def add_child(self, node):
        self.children.append(node)
        return self.children[-1]

    def remove_child(self, node):
        self.children.remove(node)

    def nth_child(self, n):
        return self.children[n]

class Tree:
    def __init__(self):
        self.root = Node(-1, 0, 0)

    def create(self, nb_tasks, tasks_lengths, eval_inf):
        #print(nb_tasks)
        self.create_impl([x for x in range(nb_tasks)], tasks_lengths, eval_inf, self.root, (0, 0, 0))

    
    def create_impl(self, tasks_list, tasks_lengths, eval_inf, current_node, current_dates):
        for i in tasks_list:
            #print("task ", i)
            d_k_a = tasks_lengths['A'][i]
            d_k_b = tasks_lengths['B'][i]
            d_k_c = tasks_lengths['C'][i]
            
            new_dates = update_dates(current_dates, (d_k_a, d_k_b, d_k_c))
            *t, t_k_c = new_dates
            #print(new_dates)
            
            new_tasks = list(tasks_list)
            new_tasks.remove(i)
            
            lower_bound = eval_inf(new_tasks, tasks_lengths, new_dates)

            new_node = current_node.add_child(Node(i, t_k_c, lower_bound, upper_bound))

            self.create_impl(new_tasks, tasks_lengths, eval_inf, eval_sup, new_node, new_dates)

    def eval(self, tasks_lengths):
        return self.eval_node(tasks_lengths, self.root)

    def eval_node(self, tasks_lengths, node):
        best_solution_value = None
        
        if len(node.children) == 0:
            return node.val

        for child in node.children:
            if best_solution_value is None:
                best_solution_value = self.eval_node(tasks_lengths, child)
            elif child.inf <= best_solution_value:
                best_solution_value = min(best_solution_value, self.eval_node(tasks_lengths, child))

        return best_solution_value
        

    def count_children(self, node, k):
        print(str(k) + " : " + str(node.inf) + " : " + str(node.val))
        for child in node.children :
            self.count_children(child, k + 1)
 
#############################################################################################
#                                                                                           #
#                                      SECOND TREE IMPLEMENTATION                           #
#                                                                                           #
#############################################################################################           

# This is the final implementation of the tree, using and implicit structure.
# There is two exact evaluation algorithms :
#   -eval : this algorithm is akin to what was implemented in the first tree implementation.
# It does not, however, need the underlying node structure to work.
#   -eval2: this algorithm is essentially the same as the first one, but trying for k steps to
# find the best path by taking the path with the best lower bound. It is rarely slower than the first
# for small k, and almost everytime a lot faster.
class Tree_Opt:
    def eval(self, nb_tasks, tasks_lengths, eval_inf):
        return self.eval_node(tasks_lengths, list(range(nb_tasks)), (0, 0, 0), eval_inf)
    
    def eval2(self, nb_tasks, tasks_lengths, eval_inf, k):
        return self.eval2_first_step(tasks_lengths, list(range(nb_tasks)), (0, 0, 0), eval_inf, k)
    
    
    def find_best_heuristic(self, tasks_lengths, task_list, current_dates, eval_inf, k):
        min_bound = math.inf

        first_task = 0
        first_task_dates = None
        for t in task_list:
            d_k_a = tasks_lengths['A'][t]
            d_k_b = tasks_lengths['B'][t]
            d_k_c = tasks_lengths['C'][t]
            
            task_list.remove(t)

            new_dates = update_dates(current_dates, (d_k_a, d_k_b, d_k_c))
            inf = eval_inf(task_list, tasks_lengths, new_dates)
            
            if inf < min_bound:
                min_bound = inf
                first_task = t
                first_task_dates = new_dates
            
            task_list.append(t)
            
        task_list.remove(first_task)

        if (k-1) == 0:
            task_list.insert(0, first_task)
            return task_list
        else:
            return [first_task] + self.find_best_heuristic(tasks_lengths, task_list, first_task_dates, eval_inf, k - 1)
    # TODO : test if k is valid (< nb tasks, > 0)        
    def eval2_first_step(self, tasks_lengths, task_list, current_dates, eval_inf, k):
        task_list = self.find_best_heuristic(tasks_lengths, task_list, current_dates, eval_inf, k)
        
        return self.eval_node(tasks_lengths, task_list, current_dates, eval_inf)
    
    def eval_node(self, tasks_lengths, task_list, current_dates, eval_inf):
        best_solution_value = None

        if len(task_list) == 0:
            *t, end = current_dates
            return (end, [])

        new_list = list(task_list)
        best_task = 0
        best_ord = None
        for t in task_list:
            d_k_a = tasks_lengths['A'][t]
            d_k_b = tasks_lengths['B'][t]
            d_k_c = tasks_lengths['C'][t]
            
            new_list.remove(t)

            new_dates = update_dates(current_dates, (d_k_a, d_k_b, d_k_c))
            inf = eval_inf(new_list, tasks_lengths, new_dates)
            
            if best_solution_value is None:
                best_solution_value, best_ord = self.eval_node(tasks_lengths, new_list, new_dates, eval_inf)
                best_task = t
            elif inf <= best_solution_value:
                solution_value, current_ord = self.eval_node(tasks_lengths, new_list, new_dates, eval_inf)
                if best_solution_value > solution_value:
                    best_solution_value = solution_value
                    best_ord = current_ord
                    best_task = t
            new_list.append(t)

        return (best_solution_value, [best_task] + best_ord)
    
    
    def probe_LDS(self, tasks_lengths, task_list, current_dates, eval_inf, k, dates):
        if task_list == []:
            return (current_dates, [])
        if k == 0:
            new_list = self.find_best_heuristic(tasks_lengths, task_list, current_dates, eval_inf, 1)
            best_task = new_list[0]
            new_list = new_list[1:]
            new_dates = update_dates(current_dates, (tasks_lengths['A'][best_task], tasks_lengths['B'][best_task], tasks_lengths['C'][best_task]))
            dates, best_ord = self.probe_LDS(tasks_lengths, new_list, new_dates, eval_inf, 0, dates)
            return dates, [best_task] + best_ord
        else:
            new_list = list(task_list)
            if dates:
                *d, best_solution_value = dates
            else:
                best_solution_value = math.inf
                
            best_dates = None
            best_ord = None
            best_task = 0
            intermediate_res = 0
            for t in task_list:
                d_k_a = tasks_lengths['A'][t]
                d_k_b = tasks_lengths['B'][t]
                d_k_c = tasks_lengths['C'][t]
                
                new_list.remove(t)
                
                new_dates = update_dates(current_dates, (d_k_a, d_k_b, d_k_c))
                inf = eval_inf(new_list, tasks_lengths, new_dates)

                if inf <= best_solution_value:
                    new_dates, new_ord = self.probe_LDS(tasks_lengths, new_list, new_dates, eval_inf, k - 1, dates)
                    if new_dates:
                        *d, intermediate_res = new_dates
                    if intermediate_res < best_solution_value:
                        best_solution_value = intermediate_res
                        best_dates = new_dates
                        best_ord = new_ord
                        best_task = t
                    
                new_list.append(t)
                
            if not best_ord:
                return (best_dates, [])
                    
        return (best_dates, [best_task] + best_ord)
        
    
    def eval_LDS(self, nb_tasks, tasks_lengths, eval_inf, k):
        dates = None
        task_list = list(range(nb_tasks))
        current_res = math.inf
        current_ord = None
        for i in range(k):
            new_ord = None
            probe_res = self.probe_LDS(tasks_lengths, task_list, (0, 0, 0), eval_inf, i, dates)
            if probe_res:
                dates, new_ord = probe_res
                if dates:
                    *d, res = dates
                print(res)
            if current_res > res:
                current_res = res
                current_ord = new_ord
            
        return (current_res, current_ord)
    
    def eval_greed_path(self, tasks_lengths, task_list, current_dates):
        new_list = list(task_list)
        new_dates = None
        
        for t in task_list:
            d_k_a = tasks_lengths['A'][t]
            d_k_b = tasks_lengths['B'][t]
            d_k_c = tasks_lengths['C'][t]
            
            new_list.remove(t)
            new_dates = update_dates(current_dates, (d_k_a, d_k_b, d_k_c))
            
        return new_dates

        
    def eval_greed(self, nb_tasks, tasks_lengths):
        task_list = list(range(nb_tasks))
        current_dates = (0, 0, 0)
        current_ord = []
        
        for i in range(nb_tasks):
            current_best_path_value = math.inf
            current_best_branch = 0
            for t in task_list:
                *d, res = self.eval_greed_path(tasks_lengths, task_list, current_dates)
                if res < current_best_path_value:
                    current_best_path_value = res
                    current_best_branch = t
            task_list.remove(current_best_branch)
            current_ord = current_ord + [current_best_branch]
            
            d_k_a = tasks_lengths['A'][current_best_branch]
            d_k_b = tasks_lengths['B'][current_best_branch]
            d_k_c = tasks_lengths['C'][current_best_branch]
            current_dates = update_dates(current_dates, (d_k_a, d_k_b, d_k_c))
        
        *d, end = current_dates
        return (end, current_ord)
        #return self.eval_node(tasks_lengths, self.root, [x for x in range(nb_tasks)])


##############################################################################################
        
# The following functions are used to compute the lower bound b1 as described in the subject
def eval_b_pi_a(tasks_list, tasks_lengths, current_dates):
    t_pi_a, t_pi_b, t_pi_c = current_dates

    b_pi_a = t_pi_a
    min_db_dc = None
    #print(tasks_list)
    for i in tasks_list:
        b_pi_a += tasks_lengths['A'][i]
        #print("sum : ", i)
        #print(min_db_dc)
        if not min_db_dc is None:
            min_db_dc = min(min_db_dc, tasks_lengths['B'][i] +  tasks_lengths['C'][i])
        else:
            min_db_dc = tasks_lengths['B'][i] +  tasks_lengths['C'][i]
    
    if min_db_dc is None:
        min_db_dc = 0

    #print(current_dates)
    #print("sum : ", (b_pi_a - t_pi_a), " min : ", min_db_dc)
    return (b_pi_a + min_db_dc)

def eval_b_pi_b(tasks_list, tasks_lengths, current_dates):
    t_pi_a, t_pi_b, t_pi_c = current_dates

    b_pi_b = t_pi_b
    min_dc = None
    for i in tasks_list:
        b_pi_b += tasks_lengths['B'][i]
        
        if not min_dc is None:
            min_dc = min(min_dc, tasks_lengths['C'][i])
        else:
            min_dc = tasks_lengths['C'][i]

    if min_dc is None:
        min_dc = 0

    return (b_pi_b + min_dc)

def eval_b_pi_c(tasks_list, tasks_lengths, current_dates):
    t_pi_a, t_pi_b, t_pi_c = current_dates

    b_pi_c = t_pi_c
    for i in tasks_list:
        b_pi_c += tasks_lengths['C'][i]

    return b_pi_c

def eval_inf_b1(tasks_list, tasks_lengths, current_dates):
    b_pi_a = eval_b_pi_a(tasks_list, tasks_lengths, current_dates)
    b_pi_b = eval_b_pi_b(tasks_list, tasks_lengths, current_dates)
    b_pi_c = eval_b_pi_c(tasks_list, tasks_lengths, current_dates)

    #print(b_pi_a, b_pi_b, b_pi_c)
    return max(b_pi_a, b_pi_b, b_pi_c)

# The following functions are used to compute the lower bound b2 as described in the subject    
def eval_inf_b2_min_sum(tasks_list, tasks_lengths, k):
    min_sum = 0

    for t in tasks_list:
        if k != t:
            min_sum += min(tasks_lengths['A'][t], tasks_lengths['C'][t])
            
    return min_sum + tasks_lengths['A'][k] + tasks_lengths['B'][k] + tasks_lengths['C'][k]

def eval_inf_b2(tasks_list, tasks_lengths, current_dates):
    t_pi_a, t_pi_b, t_pi_c = current_dates
    
    if len(tasks_list) == 0:
        return t_pi_a
    
    max_t = -math.inf
    for t in tasks_list:
        inf = eval_inf_b2_min_sum(tasks_list, tasks_lengths, t)
        if inf > max_t:
            max_t = inf
            
    return (t_pi_a + max_t)


# The following functions are used to compute the lower bound b3 as described in the subject
def eval_inf_b3_min_sum(tasks_list, tasks_lengths, k):
    min_sum = 0

    for t in tasks_list:
        if k != t:
            min_sum += min(tasks_lengths['B'][t], tasks_lengths['C'][t])
            
    return min_sum + tasks_lengths['B'][k] + tasks_lengths['C'][k]

def eval_inf_b3(tasks_list, tasks_lengths, current_dates):
    t_pi_a, t_pi_b, t_pi_c = current_dates
    
    if len(tasks_list) == 0:
        return t_pi_b
    
    max_t = -math.inf
    for t in tasks_list:
        inf = eval_inf_b3_min_sum(tasks_list, tasks_lengths, t)
        if inf > max_t:
            max_t = inf
            
    return (t_pi_b + max_t)


# This function returns the max of the three previously described lower bound, hence giving what we hope
# is the best lower bound.
def eval_inf_max(tasks_list, tasks_lengths, current_dates):
    return max(eval_inf_b1(tasks_list, tasks_lengths, current_dates),
               eval_inf_b2(tasks_list, tasks_lengths, current_dates),
               eval_inf_b3(tasks_list, tasks_lengths, current_dates))

# Approximated lower bounds (not technically lower bounds, there's no guarantee of that)
# We could parameterize them, but right know the algorithm use an alpha = 2
# alpha * eval_orig(list/alpha)
def eval_inf_b1_approx(tasks_list, tasks_lengths, current_dates):
    return 2 * eval_inf_b1(tasks_list[0:int(len(tasks_list)/2)], tasks_lengths, current_dates)    

def eval_inf_b2_approx(tasks_list, tasks_lengths, current_dates):
    return 2 * eval_inf_b2(tasks_list[0:int(len(tasks_list)/2)], tasks_lengths, current_dates)

def eval_inf_b3_approx(tasks_list, tasks_lengths, current_dates):
    return 2 * eval_inf_b3(tasks_list[0:int(len(tasks_list)/2)], tasks_lengths, current_dates)

def eval_inf_max_approx(tasks_list, tasks_lengths, current_dates):
    return max(eval_inf_b1_approx(tasks_list, tasks_lengths, current_dates),
               eval_inf_b2_approx(tasks_list, tasks_lengths, current_dates),
               eval_inf_b3_approx(tasks_list, tasks_lengths, current_dates))

# Generates non-correlated data instances. 
def generate_non_correlated_instance(nb_tasks):
    tasks_lengths = {'A':[], 'B':[], 'C':[]}
    
    for t in range(nb_tasks):
        tasks_lengths['A'].append(int(random.uniform(1, 100)))
        tasks_lengths['B'].append(int(random.uniform(1, 100)))
        tasks_lengths['C'].append(int(random.uniform(1, 100)))
        
    return tasks_lengths

# Generates time correlated data instances.
def generate_time_correlated_instance(nb_tasks):
    tasks_lengths = {'A':[], 'B':[], 'C':[]}
    
    for t in range(nb_tasks):
        base_time = random.uniform(0, 4)
        
        ai = 20 * base_time
        bi = 20 * base_time + 20
        
        tasks_lengths['A'].append(int(random.uniform(ai, bi)))
        tasks_lengths['B'].append(int(random.uniform(ai, bi)))
        tasks_lengths['C'].append(int(random.uniform(ai, bi)))

    return tasks_lengths

# Generates machine correlated data instances.
def generate_machine_correlated_instance(nb_tasks):
    tasks_lengths = {'A':[], 'B':[], 'C':[]}

    ai_val = lambda x: 15 * (x - 1) + 1
    bi_val = lambda x: 15 * (x - 1) + 100
    
    for t in range(nb_tasks):
        tasks_lengths['A'].append(int(random.uniform(ai_val(1), bi_val(1))))
        tasks_lengths['B'].append(int(random.uniform(ai_val(2), bi_val(2))))
        tasks_lengths['C'].append(int(random.uniform(ai_val(3), bi_val(3))))
        
    return tasks_lengths

nb_t, inpt = read_input("test_opt357")
nb_t = 10
tree = Tree_Opt()
inpt = generate_non_correlated_instance(nb_t)
#i#npt = {'A': [97, 52, 75, 93, 86, 60, 18, 17, 40, 1], 'B': [77, 41, 80, 48, 2, 68, 11, 63, 83, 59], 'C': [47, 18, 6, 27, 51, 54, 4, 87, 50, 60]}
#inpt={'A': [67, 8, 39, 90, 48, 52, 63, 12], 'B': [38, 82, 12, 53, 61, 9, 66, 56], 'C': [75, 45, 40, 30, 13, 4, 99, 15]}
#inpt={'A': [91, 97, 79, 35, 35, 93, 54, 97], 'B': [89, 97, 38, 93, 99, 54, 7, 39], 'C': [78, 36, 29, 57, 38, 39, 31, 99]}
print(inpt)

#print(inpt)
#tree.create(nb_t, inpt, eval_inf_max, (lambda x, y, z : math.inf))
#tree.count_children(tree.root, 0)

eval_inf = eval_inf_max

ax = plt.figure().gca()
#...

# Lower bounds benchmark (non correlated)
res_eval_b1 = []
res_eval_b2 = []
res_eval_b3 = []
res_eval_max = []
for nb_t in range(3, 10):
    res_eval_b1.append(0)
    res_eval_b2.append(0)
    res_eval_b3.append(0)
    res_eval_max.append(0)
    a = 100
    if nb_t == 9:
        a = 10
    if nb_t >= 10:
        a = 4
    for i in range(1, a):
        inpt = generate_machine_correlated_instance(nb_t)
        beg = time.time()
        tree.eval(nb_t, inpt, eval_inf_b1)
        end = time.time()
        res_eval_b1[nb_t - 3] += end - beg
        
        beg = time.time()
        tree.eval(nb_t, inpt, eval_inf_b2)
        end = time.time()
        res_eval_b2[nb_t - 3] += end - beg
        
        beg = time.time()
        tree.eval(nb_t, inpt, eval_inf_b3)
        end = time.time()
        res_eval_b3[nb_t - 3] += end - beg
        
        beg = time.time()
        tree.eval(nb_t, inpt, eval_inf_max)
        end = time.time()
        res_eval_max[nb_t - 3] += end - beg

    
    res_eval_b1[nb_t - 3] /= a
    res_eval_b2[nb_t - 3] /= a 
    res_eval_max[nb_t - 3] /= a
    res_eval_b3[nb_t - 3] /= a
    
    print("pass", nb_t)
    
plt.figure(0)
ax.set_xlabel("taille de l'instance")
ax.set_ylabel("temps d'exécution moyen (en secondes)")
x = list(range(3, 10))
ax.legend(handles=[mpatches.Patch(color='red', label="borne b1"),
                    mpatches.Patch(color="blue", label="borne b2"),
                    mpatches.Patch(color="yellow", label="borne b3"),
                    mpatches.Patch(color='green', label="borne max")])
ax.plot(x, res_eval_b1, "r")
ax.plot(x, res_eval_b2, "b")
ax.plot(x, res_eval_b3, "y")
ax.plot(x, res_eval_max, "g")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

    

beg = time.time()
res1, res2 = tree.eval(nb_t, inpt, eval_inf)
print(res1, res2, get_ord_end(res2, inpt), tree.k)
end = time.time()
print("Algo 1 : ", end - beg)
#beg = time.time()
#print(tree.eval(nb_t, inpt, eval_inf_b1))
#end = time.time()
#print("Algo 2 : ", end - beg)
#beg = time.time()
#res1, res2= tree.eval2(nb_t, inpt, eval_inf, 1)
#print(res1, res2, get_ord_end(res2, inpt))
#end = time.time()
#print("Algo 2 : ", end - beg)
#beg = time.time()
#print(tree.eval_LDS(nb_t, inpt, eval_inf, 2))
#end = time.time()
#print("Algo 3 : ", end - beg)
#beg = time.time()
#print(tree.eval_LDS(nb_t, inpt, eval_inf_b1, 2))
#end = time.time()
#print("Algo 4 : ", end - beg)
#beg = time.time()
#print(tree.eval_greed(nb_t, inpt))
#end = time.time()
#print("Algo 4 : ", end - beg)
#beg = time.time()
#print(johnson(nb_t, inpt))
#end = time.time()
#print("Algo 4 : ", end - beg)