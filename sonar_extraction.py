## Setting Extractor Function

# Note that we omit the derived settings range in centimetres and data points per scan as well as ones not
# relevant to our purposes such as speed of sound and head offset.

import os
import numpy as np


def experiment_loader(directory_name):
    root_dir = "C:\\Users\\the_n\\Documents\\PhD Stuff\\Sonar\\Data\\Demo files\\EXPERIMENTS"
    src_dir = root_dir + "\\" + directory_name
    data_files = {}
    scans = os.listdir(src_dir)
    for name in scans:
        file = open(src_dir + "\\" + name, "rb")
        data_files[name] = list(np.fromfile(src_dir + "\\" + name, dtype="ubyte"))
        file.close()
    return data_files


def file_settings(file):
    setting_changes = [i for i in range(len(file)) if file[i:i + 9] == [0, 0, 0, 0, 6, 0, 0, 0, 1]]
    settings = {"Sonar Freq": None,
                "Resolution": None,
                "Scan Speed": None,
                "Scan Mode": None,
                "Range": None,
                "Percentage Gain": None,
                "Sector Centre": None,
                "Sector Width": None}
    speeds = {0: "Super Fast", 1: "Fast", 2: "Normal", 3: "High Resolution"}
    modes = {0: "Stop", 1: "Scan", 3: "Rotate"}

    for j in range(len(setting_changes)):
        if j == 0:
            if len(setting_changes) == 1:
                print("FILE SETTINGS")
                print("-------------")
            else:
                print("INITIAL SETTINGS")
                print("----------------")
        else:
            print("\nSETTING CHANGE", j)
            print("--------------")

        if settings["Sonar Freq"] != [file[setting_changes[j] + 221] * 256 + file[setting_changes[j] + 220], "Hz"]:
            settings["Sonar Freq"] = [file[setting_changes[j] + 221] * 256 + file[setting_changes[j] + 220], "Hz"]
            print("Sonar Freq: ", settings["Sonar Freq"][0], settings["Sonar Freq"][1])
        if settings["Resolution"] != [file[setting_changes[j] + 231], "Bit"]:
            settings["Resolution"] = [file[setting_changes[j] + 231], "Bit"]
            print("Resolution: ", settings["Resolution"][0], settings["Resolution"][1])
        if settings["Scan Speed"] != [speeds[file[setting_changes[j] + 232]], ""]:
            settings["Scan Speed"] = [speeds[file[setting_changes[j] + 232]], ""]
            print("Scan Speed: ", settings["Scan Speed"][0], settings["Scan Speed"][1])
        if settings["Scan Mode"] != [modes[file[setting_changes[j] + 233]], ""]:
            settings["Scan Mode"] = [modes[file[setting_changes[j] + 233]], ""]
            print("Scan Mode: ", settings["Scan Mode"][0], settings["Scan Mode"][1])
        if settings["Range"] != [file[setting_changes[j] + 236], "m"]:
            settings["Range"] = [file[setting_changes[j] + 236], "m"]
            print("Range: ", settings["Range"][0], settings["Range"][1])
        if settings["Percentage Gain"] != [file[setting_changes[j] + 238], ""]:
            settings["Percentage Gain"] = [file[setting_changes[j] + 238], ""]
            print("Percentage Gain: ", settings["Percentage Gain"][0], settings["Percentage Gain"][1])
        if settings["Sector Centre"] != [file[setting_changes[j] + 249] * 256 + file[setting_changes[j] + 248], "deg"]:
            settings["Sector Centre"] = [file[setting_changes[j] + 249] * 256 + file[setting_changes[j] + 248], "deg"]
            print("Sector Centre: ", settings["Sector Centre"][0], settings["Sector Centre"][1])
        if settings["Sector Width"] != [file[setting_changes[j] + 251] * 256 + file[setting_changes[j] + 250], "deg"]:
            settings["Sector Width"] = [file[setting_changes[j] + 251] * 256 + file[setting_changes[j] + 250], "deg"]
            print("Sector Width: ", settings["Sector Width"][0], settings["Sector Width"][1])


def print_stats(data):
    print("Sonar Freq: ", data[247] * 256 + data[246], "Hz")
    print("Resolution: ", data[257], "Bit")
    speeds = {0: "Super Fast", 1: "Fast", 2: "Normal", 3: "High Resolution"}
    print("Scan Speed: ", speeds[data[258]])
    print("Data Points per Scan: ", data[269] * 256 + data[268])
    modes = {0: "Stop", 1: "Scan", 3: "Rotate"}
    print("Scan Mode: ", modes[data[259]])
    print("Range in m: ", data[262])
    print("Range in cm: ", data[281] * 256 + data[280])
    print("Percentage Gain: ", data[264])
    print("Start Angle: ", data[275] * 256 + data[274], "deg")
    print("Head Offset: ", 256 * data[271] + data[270], "deg")
    print("Speed of Sound: ", data[273] * 256 + data[272], "m/s")


# FUNCTION TO EXTRACT MODE AND DATA SLICES FROM FILE

def data_slices(data, *args):
    data = data[26:]  # Drop file header
    setting_changes = [i for i in range(len(data)) if
                       data[i:i + 9] == [0, 0, 0, 0, 6, 0, 0, 0, 1]]  # Find all setting changes
    data = [data[setting_changes[j]:setting_changes[j + 1]] for j in range(len(setting_changes) - 1)] + [
        data[setting_changes[-1]:]]
    modes = [d[233] for d in data]

    # Check Mode Compatability and Print Scan Type

    for m in modes:
        if m != modes[0]:
            print("This file changes scan mode midway. There may be problems with the image outputs.")
            return None
    if modes[0] == 1:
        print("Scan Mode")
    elif modes[0] == 3:
        print("Rotate Mode")
    elif modes[0] == 0:
        print("Stop Mode")
    else:
        print("Mode Not Recognised!")

    data_slices = []
    initial_time = data[0][278] + data[0][279] * 256
    turning_points = []
    while data != []:
        period = data[-1][243] * 256 + data[-1][242] + 16
        scan_centre = 1024 * (data[-1][249] * 256 + data[-1][248]) // 360 + 1
        scan_width = 1024 * (data[-1][251] * 256 + data[-1][250]) // 360
        lo_lim = (scan_centre - scan_width / 2) % 1024
        hi_lim = (scan_centre + scan_width / 2) % 1024
        slices = (len(data[-1]) - 274) // period
        data_slices += [[data[-1][282 + s * period] + data[-1][283 + s * period] * 256] +
                        data[-1][290 + s * period:274 + (s + 1) * period] for s in range(slices)]
        del data[-1]  # We delete as we go to save memory

    return modes[0], data_slices


def preprocess_data_slices(data_slices):
    num_slices = len(data_slices)
    processed_slices = []
    i = 0
    while i < num_slices - 1:
        if data_slices[i][0] != data_slices[i + 1][0]:
            processed_slices.append(data_slices[i])
            i += 1
            if i + 1 == num_slices:
                processed_slices.append(data_slices[-1])
                break

        else:
            statics = [data_slices[i], data_slices[i + 1]]
            i += 1
            if i + 1 == num_slices:
                average = [int(round(sum([statics[j][k] for j in range(len(statics))]) / len(statics))) for k in
                           range(len(statics[0]))]
                processed_slices.append(average)
            while i < num_slices - 1:
                if data_slices[i][0] == data_slices[i + 1][0]:
                    statics.append(data_slices[i + 1])
                    i += 1
                else:
                    i += 1
                    average = [int(round(sum([statics[j][k] for j in range(len(statics))]) / len(statics))) for k in
                               range(len(statics[0]))]
                    processed_slices.append(average)
                    if i + 1 == num_slices:
                        processed_slices.append(data_slices[i])
                    break
    # contraction_factor=len(processed_slices)/len(data_slices)
    # print("Contraction Factor:", contraction_factor)

    return processed_slices


# FUNCTION FOR DIRECTION IN A GIVEN BASE

# To give the correct answer of whether the scanner is moving forwards or backwards, we must account for working in base 1024. We
# define a maximum permissable jump distance in any direction, say 90 degrees or 256 in our base 1024 system. An output of 1
# means forward, an output of -1 means backwards and an output of zero will mean the numbers are the same and will give an
# error message.

def forb(a, b, jump):
    m = b - a
    if abs(m) > 256:
        m *= -1
    if m > 0:
        return 1
    elif m < 0:
        return -1
    else:
        return 0  # This is not a moving sequence


# BLOCK EXTRACTOR FUNCTION

def block_extractor(data_slices, mode):
    count = len(data_slices)
    if mode == 3:
        pass
    elif mode == 0:
        pass
    elif mode == 1:
        blocks = [[data_slices[0]]]  # We create the first entry of the first block using our first data slice.
        block = 0  # Set the index of the current increasing/decreasing data block
        i = 0  # We set the index of thr data slice we are looking at
        direction = forb(data_slices[0][0], data_slices[1][0], 256)  # Find initial direction
        blocks[0].append(data_slices[1])  # Add second data slice
        i += 1  # Position to find third slice
        new_direction = forb(data_slices[i][0], data_slices[i + 1][0], 256)  # Check direction of next slice
        while i + 2 < count:  # While we still have data slices remaining check the direction of the next slice
            new_direction = forb(data_slices[i][0], data_slices[i + 1][0], 256)  # Check direction of next slice
            while new_direction == direction:  # If new direction is same as old
                blocks[block].append(data_slices[i + 1])  # we add the next element
                i += 1  # then tell the algorithm where to check next
                if i + 2 == count:  # brief check for end of data stream
                    break
                new_direction = forb(data_slices[i][0], data_slices[i + 1][0], 256)  # Check direction of next slice
            direction *= -1  # Change direction if the old direction is not the same as the old
            block += 1  # When direction changes, we add a new block and populate it with the last two slices
            blocks.append([])
            blocks[block].append(data_slices[i])
            blocks[block].append(
                data_slices[i + 1])  # The new slice must have at least two entries to determine direction
            i += 1
        # WE NEED TO DO THE LAST TWO POINTS SEPERATELY
        return blocks

    else:
        print("MODE NOT RECOGNISED!")

def flip_merge(test):
    current=0                            # Current index
    new=test.copy()                      # Create a shallow copy of the list
    print("Initial input",new)
    if len(new[current])<10:             # We take the first two sublists seperately as we must establish a canonical direction
        if len(new[current+1])<10:       #If the first sublist is short, the second sublist establishes canonical direction.
            print("No Canonical Direction")   # If the first two lists are very short, we delete the initial list and try
            del new[current]                  # the algorithm on the remaining list. We could add in the deleted portion later.
            new=flip_merge[new]
        else:
            new[current]=sorted(new[current]+new[current+1]) # Merge and sort the first two sublists
            if new[current+1][0]>new[current+1][-1]: # If the second list is descending, then reverse the merged list as
                new[current].reverse()               # the default sorted list will be in increasing order
            del new[current+1]
    print("After first wedge",new)
    while current<(len(new)-1):                     # Now a canonical direction has been established, we can perform an
        if len(new[current])<10:                    # ordering algorithm on the rest of the sublists.
            new[current]=sorted(new[current-1]+new[current]+new[current+1]) # The previous sublist will always be in canonical
            if new[current-1][0]>new[current-1][-1]:                # As a consequence of the previous code. If the current
                new[current].reverse()                              # sublist is small, we give it the correct ordering.
            del new[current-1]
            del new[current]                     # If the above is true, we merge 3 sublists into 1 so our position must be altered
            current-=2
        current+=1
    print(len(new))
    if len(new[current])<10:                     # For the last entries, there are only 2 sublists to merge
        new[current]=sorted(new[current-1]+new[current])
        if new[current-1][0]>new[current-1][-1]:
                new[current].reverse()
        del new[current-1]
    print("Final Output",new)
    return new

# For any two consecutive integers in the sequence, this gives 1 if moving "forwards" and -1 if moving "backwards".
# Direction is based on the shortest path if we imagine a clock numbered with integers up to the base.

def dir_in_b(a,b,base):
    if a==b:
        print("This is not a moving sequence")
        return 0
    else:
        m=(a-b)%base-(b-a)%base
        if m>0:
            return 1
        elif m<0:
            return -1
        else:
            return 0

def sort_mod(sorted_list, jump):
    current = 0
    start = None
    while start == None:
        if sorted_list[current + 1] - sorted_list[current] > jump:
            start = current + 1
        else:
            current += 1
            if current == len(sorted_list) - 1:
                break
    if start == None:
        return sorted_list
    else:
        return sorted_list[start:] + sorted_list[:start]

# SORTING LISTS IN MODULAR NUMBER SYSTEM WITH MAX PERMISSABLE JUMP

def sort_lists_mod(lists,jump):
    sorted_lists=sorted(lists)
    current=0
    start=None
    while start==None:
        if sorted_lists[current+1][0]-sorted_lists[current][0]>jump:
            start=current+1
        else:
            current+=1
            if current==len(sorted_lists)-1:
                break
    if start==None:
        return sorted_lists
    else:
        return sorted_lists[start:]+sorted_lists[:start]

# MAIN FLIP-MERGE FOR PREPROCESSED DATA SLICES

def flip_merge_data(test_data):
    current=0                            # Current index
    new=test_data.copy()                 # Create a shallow copy of the list
    if len(new[current])<10:             # We take the first two sublists seperately as we must establish a canonical direction
        if len(new[current+1])<10:       #If the first sublist is short, the second sublist establishes canonical direction.
            print(new[current],new[current+1])
            print("No Canonical Direction")   # If the first two lists are very short, we delete the initial list and try
            del new[current]                  # the algorithm on the remaining list. We could add in the deleted portion later.
            new=flip_merge(new)
        else:
            new[current]=sort_lists_mod(sorted(new[current]+new[current+1]),40) # Merge and sort the first two sublists
            if dir_in_b(new[current+1][0][0],new[current+1][-1][0],1024)==-1: # If the second list is descending, then reverse the merged list as
                new[current].reverse()               # the default sorted list will be in increasing order
            del new[current+1]
    while current<(len(new)-1):                     # Now a canonical direction has been established, we can perform an
        if len(new[current])<10:                    # ordering algorithm on the rest of the sublists.
            new[current]=sort_lists_mod(sorted(new[current-1]+new[current]+new[current+1]),40) # The previous sublist will always be in canonical
            if dir_in_b(new[current+1][0][0],new[current+1][-1][0],1024)==-1:                # As a consequence of the previous code. If the current
                new[current].reverse()                              # sublist is small, we give it the correct ordering.
            del new[current-1]
            del new[current]                     # If the above is true, we merge 3 sublists into 1 so our position must be altered
            current-=2
        current+=1
    if len(new[current])<10:                     # For the last entries, there are only 2 sublists to merge
        new[current]=sort_lists_mod(sorted(new[current-1]+new[current]),40)
        if new[current-1][0]>new[current-1][-1]:
                new[current].reverse()
        del new[current-1]
    return [preprocess_data_slices(new[i]) for i in range(len(new))]


def flip_anticlockwise(data_blocks):
    for b in range(len(data_blocks)):
        if forb(data_blocks[b][0][0],data_blocks[b][-1][0],40)<0:
            data_blocks[b]=data_blocks[b][::-1]
    return data_blocks

def get_limits(data_blocks):
    limit_list=sorted([b[0][0] for b in data_blocks]+[b[-1][0] for b in data_blocks])
    limit_list=sort_mod(limit_list,200)
    return [limit_list[0],limit_list[-1]]

from random import normalvariate as norm
from random import random as rand, randint

def block_generator(blocks, depth, limits, sparsity, jitter, empty):
    output_blocks = []
    for b in range(blocks):
        output_blocks.append([])
        b_limits = (int(norm(limits[0], jitter)), int(norm(limits[1], jitter)))
        current = b_limits[0]
        output_blocks[b].append([b_limits[0]] + [10 if rand() < empty else randint(11, 127) for i in range(depth)])
        while forb(current, b_limits[1], 40) >= 0:  # While we have not passed the upper limit
            current = (current + (sparsity + 1)) % 1024
            output_blocks[b].append([current] + [10 if rand() < empty else randint(11, 127) for i in range(depth)])
        if b % 2 == 1:
            output_blocks[b] = output_blocks[b][::-1]

    return output_blocks

from scipy.interpolate import interp2d as inter
import numpy as np

def linear_inter(list1,list2,intervals):
    if len(list1)!=len(list2):
        print("Cannot be done.")
        return None
    else:
        y=[1,intervals+2]
        x=list(range(len(list1)))
        z=[list1,list2]
        f=inter(x,y,z,kind="linear")
        all_data=[list1]
        for i in range(1,intervals+1):
            all_data.append(np.round(f(x,1+i)))
        all_data.append(list2)
    return all_data

def pad(data_blocks,limits, mode):
    data_blocks=data_blocks.copy()
    depth=len(data_blocks[0][0])-1              # Number of data readings per angle
    for b in range(len(data_blocks)): # Pad at the beginning.
        if data_blocks[b][0][0]==limits[0]: # If data already starts at the lower limit, we do not add any blank blocks
            pass
        else:       # If not, we generate a block of empty data for from the start of the lower limit until the start of the data block
            pad_start=block_generator(1,depth,(limits[0],data_blocks[b][0][0]-2),0,0,1)[0][::-1]
            data_blocks[b]=data_blocks[b][::-1]+pad_start
            data_blocks[b]=data_blocks[b][::-1]
    expanded_blocks=[]
    for b in range(len(data_blocks)):
        expanded=block_generator(1,depth,(limits[0],data_blocks[b][-1][0]-2),0,0,1)[0] # Generate an expanded empty block to fill
        current=0
        if mode=="F":
            for e in range(len(expanded)):
                if data_blocks[b][current][0]==expanded[e][0]:
                    expanded[e]=data_blocks[b][current]
                    current+=1
                else:
                    expanded[e]=[expanded[e][0]]+expanded[e-1][1:]
        elif mode=="L":
            pass
        else:
            return pad(data_blocks,limits,"F")
        expanded_blocks.append(expanded)
    for e in range(len(expanded_blocks)):
        if expanded_blocks[e][-1][0]==limits[1]:
            pass
        else:
            pad_end=block_generator(1,depth,(expanded_blocks[e][-1][0]+1,limits[1]-1),0,0,1)[0]
            expanded_blocks[e]+=pad_end
    return expanded_blocks

# Converts the data to array form with the option to convert to image form and save to file
import cv2

def image_format(full_data_blocks,*args):
    for f in range(len(full_data_blocks)):
        full_data_blocks[f]=np.array(full_data_blocks[f],dtype=np.uint8)[:,1:]*2 # Converts lists into list of arrays without angle info.
    if args[0]!=None: # If path of directory given, write all images to this file
        location=args[0]
        label=args[1]
        for f in range(len(full_data_blocks)):
            cv2.imwrite(location+"\\"+label+str(f)+".jpg", full_data_blocks[f])
    return full_data_blocks

def images_from_data(file_array,*args):
    mode,unfiltered=data_slices(file_array,*args)   # Extracts the pure data from the file discarding the extra stuff we don't need
    preprocessed=preprocess_data_slices(unfiltered)  # Merges consecutive readings at same angle.
    data_blocks=block_extractor(preprocessed,mode)     # Extracts blocks from preprocessed data slices
    if mode==1:
        data_blocks=flip_merge_data(data_blocks)
        data_blocks=flip_anticlockwise(data_blocks)
        angles=[[d[0] for d in block ] for block in data_blocks]
    limits=get_limits(data_blocks)
    if forb(limits[0],limits[1],40)==-1:
        limits=limits[::-1]
    standardised_blocks=pad(data_blocks,limits, mode)
    data_blocks=image_format(standardised_blocks,*args)
    return data_blocks, angles

def process_directory(source_dir,target_dir):
    objects=experiment_loader(source_dir)
    for o in objects:
        print(o)
        data,angles=images_from_data(objects[o],target_dir, o[:-4]+" ")

