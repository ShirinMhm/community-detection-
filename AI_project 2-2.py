from operator import ne
import random
from unittest import result
from pandas import DataFrame
from sklearn.cluster import KMeans
def  find_adj():
    file1=open("sample dataset.txt","r")
    a=file1.readline()
    n=int(a)
    Adj=[[0 for x in range(n)]for y in range(n)]
    for i in range(2*n):
        x,y=file1.readline().split()
        x=int(x)
        y=int(y)
        Adj[x-1][y-1]=1
        Adj[y-1][x-1]=1
    return Adj,n
class DS:
    def __init__ (self, num):
        self.parent = []
        for i in range(num+1):
            self.parent.append(i)
    
    def find(self, node):
        return self.parent[node]
    
    def union (self, node1, node2):
        a = self.find(node1)
        b = self.find(node2)
        for i in range(len(self.parent)):
            if self.parent[i] == b:
                self.parent[i] = a
    
    def disjoint_set (self, cuc, length):
        for i in range(1, length+1):
            selected_ngh = neighbors[i][cuc.habitat[i]]
            self.union(i, selected_ngh)

    def result(self):
        result = [[] for i in range(len(self.parent))]
        for i in range(1, len(self.parent)):
            result[self.find(i)].append(i)

        j = 0
        while j < len(result):
            if result[j] == []:
                result.pop(j)
            else:
                j += 1
        return result
class cuckoo:
    def __init__(self,n,adj):
        self.habitat=[0 for i in range(n+1)]
        self.adj=adj
    def cal_Q(self,n):
        p = DS(n)
        p.disjoint_set(self, n)
        xi = 0
        result = 0
        k = [0 for i in range(n+1)]
        for i in range(n+1):
            for j in range(n+1):
                if self.adj[i][j] == 1:
                    k[i] += 1
                    xi += 1
        for i in range(1, n+1):
            for j in range(1, n+1):
                if p.find(i) == p.find(j):
                    result +=self.adj[i][j] - (k[i] * k[j] / (xi))
        result /= (xi)

        if result < 0:
            return 0
        elif result > 1:
            return 1
        return result  
    def diff (self, other, n):
        dif_sum = 0
        for i in range(n+1):
            dif = self.habitat[i] - other.habitat[i]
            if dif >= 0:
                dif_sum += dif
            else:
                dif_sum -= dif
        return dif_sum  
    def spawn (self, adj,neighbor, egg_num, ELR, n):
        egg_list = []
        while len(egg_list) <egg_num:
            new_egg = cuckoo(n)
            for i in range(1, n+1):
                new_egg.habitat[i] = self.habitat[i]
                if random.randint(0, 3) == 0:
                    change = random.randint(0, 3)
                    new_egg.habitat[i] = (new_egg.habitat[i] + change) % len(neighbor[i])

            if self.diff(new_egg,n) < ELR:
                egg_list.append(new_egg)

        return egg_list   
def find_neighbors (nodes, adj, n):
    neighbors = []
    for i in range(n+1):
        if adj[nodes][i] == 1:
            neighbors.append(i)
    return neighbors
def random_cuckoo (adj, neighbors, n):
    new = cuckoo(n)
    for i in range(1,n+1):
        neighbors = find_neighbors(i, adj, n)
        new.habitat[i] = random.randint(0, len(neighbors[i])-1)
    return new
def find_egg_and_ELR (low, high, alpha, population):
    egg_num = []
    total_num = 0
    ELR = []

    for i in range(population):
        egg_num.append(random.randint(low, high))
        total_num += egg_num[i]
    
    for i in range(population):
        ELR.append(alpha * (egg_num[i] / total_num) * (high - low))
    
    return egg_num, ELR
def merge_sort (arr1, arr2):
    if len(arr1) > 1:
        mid = len(arr1)//2
        L1 = arr1[:mid]
        L2 = arr2[:mid]
        R1 = arr1[mid:]
        R2 = arr2[mid:]
        merge_sort(L1, L2)
        merge_sort(R1, R2)
  
        i = j = k = 0
        while i < len(L1) and j < len(R1):
            if L1[i] > R1[j]:
                arr1[k] = L1[i]
                arr2[k] = L2[i]
                i += 1
            else:
                arr1[k] = R1[j]
                arr2[k] = R2[j]
                j += 1
            k += 1
  
        while i < len(L1):
            arr1[k] = L1[i]
            arr2[k] = L2[i]
            i += 1
            k += 1
  
        while j < len(R1):
            arr1[k] = R1[j]
            arr2[k] = R2[j]
            j += 1
            k += 1
def choosing_best_eggs (egg_list, population, n):
    egg_list_Q = []
    for i in range(len(egg_list)):
        egg_list_Q.append(egg_list[i].cal_Q(n))

    merge_sort(egg_list_Q, egg_list)
    egg_list = egg_list[:population]
    egg_list_Q = egg_list_Q[:population]

    return egg_list, egg_list_Q
def migration_cor (cuckoo_list, cuckoo_Q_list,neighbors, population, n):
    Xs = [""]
    coordinates = [[]]
    for i in range(1, n+1):
        Xs.append("x" + str(i))
        coordinates.append([])
    
    for cuckoo in cuckoo_list:
        for i in range(1, n+1):
            coordinates[i].append(cuckoo.habitat[i])
    
    data = {}
    for i in range(1, n+1):
        data.update({Xs[i]: coordinates[i]})
    
    df = DataFrame(data, columns=Xs[1:])
    kmeans = KMeans(n_clusters=3).fit(df)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    Q_sum = [0, 0, 0]
    for i in range(population):
        Q_sum[labels[i]] += cuckoo_Q_list[i]
    
    if Q_sum[0] > Q_sum[1] and Q_sum[0] > Q_sum[2]:
        dest_arr = centers[0]
    elif Q_sum[1] > Q_sum[2]:
        dest_arr = centers[1]
    else:
        dest_arr = centers[2]

    dest = [0]
    for i in range(1, n+1):
        dest.append(dest_arr[i-1])
        dest[i] = int(round(dest[i], 0) % len(neighbors[i]))

    return dest
def migration (cuc_list, dest, n):
    best_cuc = cuckoo(n)
    best_cuc.habitat = dest

    for cuc in cuc_list:
        diff = cuc.diff(best_cuc, n)
        prob = max(round(diff / n, 0)*2, 1)
        for i in range(n):
            if random.random() < prob:
                cuc.habitat[i] = best_cuc.habitat[i]
def main_algorithm (adj, neighbors , population , iterations , low, high, alpha,n):
    cuc_list = []
    for i in range(population):
        cuc_list.append(random_cuckoo(adj, neighbors, n))
    
    best_cuc = cuckoo(n)
    max_Q = 0

    for count in range(iterations):
        egg_num, ELR = find_egg_and_ELR (low, high, alpha, population)

        egg_list = []
        for i in range(population):
            my_egg_list = cuc_list[i].spawn(adj, neighbors, egg_num[i], ELR[i], n)
            for egg in my_egg_list:
                if random.randint(0, 2) != 0:
                    egg_list.append(egg)
        
        cuc_list_Q = []
        cuc_list, cuc_list_Q = choosing_best_eggs(egg_list, population, n)

        if cuc_list_Q[0] > max_Q:
            max_Q = cuc_list_Q[0]
            for i in range(n+1):
                best_cuc.habitat[i] = cuc_list[0].habitat[i]

        if count < iterations - 1:
            dest = migration_cor(cuc_list, cuc_list_Q,neighbors, population, n)
            migration(cuc_list, dest, n)
        
        if count % 10 == 0:
            print("number of iterations =", count, " / max value = ", max_Q)

    p = DS(n)
    p.disjoint_set(best_cuc, n)
    result = p.result()

    return result
neighbors = []
adj,n=find_adj()
for i in range(n+1):
    neighbors.append(find_neighbors(i, adj, n))
print(main_algorithm(adj,neighbors,10,100,5,8,10,n))
