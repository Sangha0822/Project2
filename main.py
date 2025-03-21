import numpy as np
import random as rand
import math as m
import time

def forward_search(file):
    data = np.loadtxt(file)
    num_features = data.shape[1]-1 #shape[1] is the size of column;
    set_of_features = []
    highest_accuracy = 0
    highest_features = []

    print("Beginning search.")

    startTime = time.time()
    for i in range(1, num_features+1):
        best_accuracy = 0
        this_level_feature = []
            
        for k in range(1, num_features+1):
            temp_space = []
            if not existing_feature(set_of_features, k):
                temp_space = set_of_features + [k]
                print(f"Using feature(s)  {temp_space}", end=" ")
                curr_accuracy = forward_accuracy(data, set_of_features, k)
                if best_accuracy < curr_accuracy:
                    best_accuracy = curr_accuracy
                    if not this_level_feature: #if it is empty, I wan to add to the list.
                        this_level_feature.append(k)
                    else:
                        this_level_feature.pop() #if it is not empty, pop and add after to maintain only one feature in there.
                        this_level_feature.append(k)
                if(highest_accuracy < best_accuracy):
                    highest_accuracy = best_accuracy
                    highest_features = set_of_features + [k]
        if this_level_feature:
            set_of_features.append(this_level_feature[0])
        print(f"Feature set {this_level_feature} was best, accuracy is {best_accuracy * 100:.1f}%")
    print(f"Finished search!! The best feature subset is {highest_features}, which has an accuracy of {highest_accuracy * 100:.1f}%")
    print("Runtime: ", round((time.time() - startTime),1)," seconds")
    

def forward_accuracy(data, set_of_features, feature_add):
    forward_temp_copy_feature(data, "temp.txt", set_of_features, feature_add)
    temp_data = np.loadtxt("temp.txt", ndmin=2)
    number_correct = 0
    num_rows = data.shape[0]
    for i in range(num_rows):
        nearest_distance = float('inf')
        nearest_location = float('inf')
        objectClassify = data[i,0]
        objectFeatures = temp_data[i, 0:]
        for k in range(num_rows):
            if k != i:
                sqrtDiff = (objectFeatures - temp_data[k, 0:]) ** 2
                distance = m.sqrt(sum(sqrtDiff))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_location = k
                    nearest_label = data[nearest_location,0]
        if objectClassify == nearest_label:
            number_correct += 1
    accuracy = number_correct / num_rows
    print(f"accuracy is {accuracy * 100:.1f}%")
    return accuracy

def forward_temp_copy_feature(data,temp_file, set_of_features, feature_add):
    temp_features = set_of_features + [feature_add]
    if not temp_features:
        return
    col_of_features = data[:,temp_features]
    with open(temp_file, "w") as temp:
        for row in col_of_features:
            val = []
            for i in row:
                val.append(str(i))
            temp.write(" ".join(val) + "\n")


def backward_search(file):
    data = np.loadtxt(file)
    num_features = data.shape[1] - 1  
    set_of_features = list(range(1, num_features + 1))  
    highest_accuracy = 0
    highest_features = set_of_features.copy()

    print("Beginning search.")

    startTime = time.time()
    for i in range(1, num_features + 1):  
        best_accuracy = 0
        feature_to_remove = None

        for k in set_of_features: 
            temp_space = set_of_features.copy()
            temp_space.remove(k) 

            print(f"Using feature(s)  {temp_space}", end=" ")
            curr_accuracy = backward_accuracy(data, set_of_features, k)

            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                feature_to_remove = k  

            if highest_accuracy < best_accuracy:
                highest_accuracy = best_accuracy
                highest_features = temp_space.copy()

        if feature_to_remove is not None:
            set_of_features.remove(feature_to_remove)
            print(f"Feature {feature_to_remove} was removed, accuracy is {best_accuracy * 100:.1f}%")

    print(f"\nFinished search! The best feature subset is {highest_features}, with accuracy {highest_accuracy * 100:.1f}%")
    print("Runtime:", round((time.time() - startTime), 1), "seconds")

def backward_accuracy(data, set_of_features, feature_add):
    backward_temp_copy_feature(data, "temp.txt", set_of_features, feature_add)
    temp_data = np.loadtxt("temp.txt", ndmin=2)
    number_correct = 0
    num_rows = data.shape[0]
    for i in range(num_rows):
        nearest_distance = float('inf')
        nearest_location = float('inf')
        objectClassify = data[i,0]
        objectFeatures = temp_data[i, 0:]
        for k in range(num_rows):
            if k != i:
                sqrtDiff = (objectFeatures - temp_data[k, 0:]) ** 2
                distance = m.sqrt(sum(sqrtDiff))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_location = k
                    nearest_label = data[nearest_location,0]
        if objectClassify == nearest_label:
            number_correct += 1
    accuracy = number_correct / num_rows
    print(f"accuracy is {accuracy * 100:.1f}%")
    return accuracy

def existing_feature(curr_feature_set, feature_add):
    return feature_add in curr_feature_set

def backward_temp_copy_feature(data,temp_file, set_of_features, feature_remove):
    temp_features = []
    for f in set_of_features:
        if f != feature_remove:
            temp_features.append(f)

    if not temp_features:
        return
    
    col_of_features = data[:,temp_features]
    with open(temp_file, "w") as temp:
        for row in col_of_features:
            val = []
            for i in row:
                val.append(str(i))
            temp.write(" ".join(val) + "\n")


#backward_search("CS170_Small_Data__107.txt")
#backward_search("CS170_Large_Data__59.txt")
forward_search("CS170_Small_Data__107.txt")
#forward_search("CS170_Large_Data__59.txt")