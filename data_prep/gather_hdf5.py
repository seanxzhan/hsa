# ADAPTED FROM
# https://github.com/czq142857/IM-NET/blob/master/point_sampling/2_gather_256vox_16_32_64.py
import numpy as np
import os
import json
import h5py
import random
from multiprocessing import Process, Queue
import queue
from tqdm import tqdm
from utils import misc, binvox_rw

orig_size = 256
comp_size = 128
batch_size_compressed = comp_size**3


def sample_point_in_cube(block, target_value, halfie):
    halfie2 = halfie*2

    for i in range(100):
        x = np.random.randint(halfie2)
        y = np.random.randint(halfie2)
        z = np.random.randint(halfie2)
        if block[x, y, z] == target_value:
            return x, y, z

    if block[halfie, halfie, halfie] == target_value:
        return halfie, halfie, halfie

    i = 1
    ind = np.unravel_index(np.argmax(block[
        halfie-i:halfie+i, halfie-i:halfie+i, halfie-i:halfie+i], axis=None),
        (i*2, i*2, i*2))
    if block[ind[0]+halfie-i,
             ind[1]+halfie-i, ind[2]+halfie-i] == target_value:
        return ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i

    for i in range(2, halfie+1):
        six = [(halfie-i, halfie, halfie), (halfie+i-1, halfie, halfie),
               (halfie, halfie, halfie-i), (halfie, halfie, halfie+i-1),
               (halfie, halfie-i, halfie), (halfie, halfie+i-1, halfie)]
        for j in range(6):
            if block[six[j]] == target_value:
                return six[j]
        ind = np.unravel_index(np.argmax(
            block[halfie-i:halfie+i, halfie-i:halfie+i, halfie-i:halfie+i],
            axis=None),  (i*2, i*2, i*2))
        if block[ind[0]+halfie-i,
                 ind[1]+halfie-i, ind[2]+halfie-i] == target_value:
            return ind[0]+halfie-i, ind[1]+halfie-i, ind[2]+halfie-i
    print('hey,  error in your code!')
    exit(0)


def get_points_from_vox(q, path, compress):
    file_count = 0
    for filename in misc.sorted_alphanumeric(os.listdir(path)):
        try:
            print(filename)
            # if file_count  ==  10:
            # 	break
            # voxel_model_64 = np.load(path + '/' + filename)
            # voxel_model_64= voxel_model_64.astype('uint8')
            with open(path + '/' + filename, 'rb') as f:
                voxel_model_88 = binvox_rw.read_as_3d_array(f)
                voxel_model_88 = voxel_model_88.data.astype(int)
        except OSError as err:
            print("error in loading\n" + err)
            exit(-1)

        if not compress:
            # print(filename)
            # print("model dimension: {}".format(voxel_model_88.shape))
            # visualize.visualize_3d_arr(voxel_model_88)

            dim = voxel_model_88.shape[0]

            orig_points_88 = np.zeros([dim**3, 3], np.uint8)
            orig_values_88 = np.zeros([dim**3, 1], np.uint8)
            orig_voxels = np.reshape(voxel_model_88, (dim, dim, dim, 1))

            counter = 0
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        orig_points_88[counter] = [i, j, k]
                        orig_values_88[counter] = voxel_model_88[i][j][k]
                        counter += 1
            assert counter == dim**3

            # name = re.search('(.*).binvox', filename).group(1)
            # name = int(name)

            q.put([file_count, orig_points_88, orig_values_88, orig_voxels])
            file_count += 1
        else:
            # EDIT: compress model 64 -> 16
            dim_voxel = comp_size
            voxel_model_temp = np.zeros(
                [dim_voxel, dim_voxel, dim_voxel], np.uint8)
            multiplier = int(orig_size/dim_voxel)
            halfie = int(multiplier/2)
            for i in range(dim_voxel):
                for j in range(dim_voxel):
                    for k in range(dim_voxel):
                        voxel_model_temp[i, j, k] = np.max(
                            voxel_model_88[
                                i*multiplier:(i+1)*multiplier,
                                j*multiplier:(j+1)*multiplier,
                                k*multiplier:(k+1)*multiplier])
            # write voxel
            # visualize.visualize_3d_arr(voxel_model_temp)
            sample_voxels = np.reshape(
                voxel_model_temp,
                (dim_voxel, dim_voxel, dim_voxel, 1))
            # sample points near surface
            batch_size = batch_size_compressed

            sample_points = np.zeros([batch_size, 3], np.uint8)
            sample_values = np.zeros([batch_size, 1], np.uint8)
            batch_size_counter = 0
            voxel_model_temp_flag = np.zeros(
                [dim_voxel, dim_voxel, dim_voxel], np.uint8)
            temp_range = list(range(1, dim_voxel-1, 4)) + \
                list(range(2, dim_voxel-1, 4)) + \
                list(range(3, dim_voxel-1, 4)) + \
                list(range(4, dim_voxel-1, 4))
            for j in temp_range:
                if (batch_size_counter >= batch_size):
                    break
                for i in temp_range:
                    if (batch_size_counter >= batch_size):
                        break
                    for k in temp_range:
                        if (batch_size_counter >= batch_size):
                            break
                        if (np.max(
                            voxel_model_temp[i-1:i+2, j-1:j+2, k-1:k+2]) !=
                                np.min(
                                voxel_model_temp[i-1:i+2, j-1:j+2, k-1:k+2])):
                            si, sj, sk = sample_point_in_cube(
                                voxel_model_88[
                                    i*multiplier:(i+1)*multiplier,
                                    j*multiplier:(j+1)*multiplier,
                                    k*multiplier:(k+1)*multiplier],
                                voxel_model_temp[i, j, k], halfie)
                            sample_points[batch_size_counter, 0] = \
                                si+i*multiplier
                            sample_points[batch_size_counter, 1] = \
                                sj+j*multiplier
                            sample_points[batch_size_counter, 2] = \
                                sk+k*multiplier
                            sample_values[batch_size_counter, 0] = \
                                voxel_model_temp[i, j, k]
                            voxel_model_temp_flag[i, j, k] = 1
                            batch_size_counter += 1
            if (batch_size_counter >= batch_size):
                print("16-- batch_size exceeded!")
                exceed_16_flag = 1
            else:
                exceed_16_flag = 0
                # fill other slots with random points
                while (batch_size_counter < batch_size):
                    while True:
                        i = random.randint(0, dim_voxel-1)
                        j = random.randint(0, dim_voxel-1)
                        k = random.randint(0, dim_voxel-1)
                        if voxel_model_temp_flag[i, j, k] != 1:
                            break
                    si, sj, sk = sample_point_in_cube(
                        voxel_model_88[
                            i*multiplier:(i+1)*multiplier,
                            j*multiplier:(j+1)*multiplier,
                            k*multiplier:(k+1)*multiplier],
                        voxel_model_temp[i, j, k], halfie)
                    sample_points[batch_size_counter, 0] = si+i*multiplier
                    sample_points[batch_size_counter, 1] = sj+j*multiplier
                    sample_points[batch_size_counter, 2] = sk+k*multiplier
                    sample_values[batch_size_counter, 0] = \
                        voxel_model_temp[i, j, k]
                    voxel_model_temp_flag[i, j, k] = 1
                    batch_size_counter += 1

            sample_points_16 = sample_points
            sample_values_16 = sample_values

            # name = re.search('(.*).npy',  filename).group(1)
            # name = int(name)
            # print(name)
            print(file_count)

            q.put([
                file_count,
                exceed_16_flag,
                sample_points_16,
                sample_values_16,
                sample_voxels])
            file_count += 1


def get_points_from_vox_multi_thread(q, path, compress, name_list):
    name_num = len(name_list)
    for idx in range(name_num):
        try:
            print(idx, '/', name_num)
            # if file_count  ==  10:
            # 	break
            # voxel_model_64 = np.load(path + '/' + filename)
            # voxel_model_64= voxel_model_64.astype('uint8')
            with open(name_list[idx][1], 'rb') as f:
                voxel_model_88 = binvox_rw.read_as_3d_array(f)
                voxel_model_88 = voxel_model_88.data.astype(int)
        except OSError as err:
            print("error in loading\n" + err)
            exit(-1)

        if not compress:
            # print(filename)
            # print("model dimension: {}".format(voxel_model_88.shape))
            # visualize.visualize_3d_arr(voxel_model_88)

            dim = voxel_model_88.shape[0]

            orig_points_88 = np.zeros([dim**3, 3], np.uint8)
            orig_values_88 = np.zeros([dim**3, 1], np.uint8)
            orig_voxels = np.reshape(voxel_model_88, (dim, dim, dim, 1))

            counter = 0
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        orig_points_88[counter] = [i, j, k]
                        orig_values_88[counter] = voxel_model_88[i][j][k]
                        counter += 1
            assert counter == dim**3

            # name = re.search('(.*).binvox', filename).group(1)
            # name = int(name)

            q.put([name_list[idx][0], orig_points_88, orig_values_88, orig_voxels])
        else:
            # EDIT: compress model 64 -> 16
            dim_voxel = comp_size
            voxel_model_temp = np.zeros(
                [dim_voxel, dim_voxel, dim_voxel], np.uint8)
            multiplier = int(orig_size/dim_voxel)
            halfie = int(multiplier/2)
            for i in range(dim_voxel):
                for j in range(dim_voxel):
                    for k in range(dim_voxel):
                        voxel_model_temp[i, j, k] = np.max(
                            voxel_model_88[
                                i*multiplier:(i+1)*multiplier,
                                j*multiplier:(j+1)*multiplier,
                                k*multiplier:(k+1)*multiplier])
            # write voxel
            # visualize.visualize_3d_arr(voxel_model_temp)
            sample_voxels = np.reshape(
                voxel_model_temp,
                (dim_voxel, dim_voxel, dim_voxel, 1))
            # sample points near surface
            batch_size = batch_size_compressed

            sample_points = np.zeros([batch_size, 3], np.uint8)
            sample_values = np.zeros([batch_size, 1], np.uint8)
            batch_size_counter = 0
            voxel_model_temp_flag = np.zeros(
                [dim_voxel, dim_voxel, dim_voxel], np.uint8)
            temp_range = list(range(1, dim_voxel-1, 4)) + \
                list(range(2, dim_voxel-1, 4)) + \
                list(range(3, dim_voxel-1, 4)) + \
                list(range(4, dim_voxel-1, 4))
            for j in temp_range:
                if (batch_size_counter >= batch_size):
                    break
                for i in temp_range:
                    if (batch_size_counter >= batch_size):
                        break
                    for k in temp_range:
                        if (batch_size_counter >= batch_size):
                            break
                        if (np.max(
                            voxel_model_temp[i-1:i+2, j-1:j+2, k-1:k+2]) !=
                                np.min(
                                voxel_model_temp[i-1:i+2, j-1:j+2, k-1:k+2])):
                            si, sj, sk = sample_point_in_cube(
                                voxel_model_88[
                                    i*multiplier:(i+1)*multiplier,
                                    j*multiplier:(j+1)*multiplier,
                                    k*multiplier:(k+1)*multiplier],
                                voxel_model_temp[i, j, k], halfie)
                            sample_points[batch_size_counter, 0] = \
                                si+i*multiplier
                            sample_points[batch_size_counter, 1] = \
                                sj+j*multiplier
                            sample_points[batch_size_counter, 2] = \
                                sk+k*multiplier
                            sample_values[batch_size_counter, 0] = \
                                voxel_model_temp[i, j, k]
                            voxel_model_temp_flag[i, j, k] = 1
                            batch_size_counter += 1
            if (batch_size_counter >= batch_size):
                print("16-- batch_size exceeded!")
                exceed_16_flag = 1
            else:
                exceed_16_flag = 0
                # fill other slots with random points
                while (batch_size_counter < batch_size):
                    while True:
                        i = random.randint(0, dim_voxel-1)
                        j = random.randint(0, dim_voxel-1)
                        k = random.randint(0, dim_voxel-1)
                        if voxel_model_temp_flag[i, j, k] != 1:
                            break
                    si, sj, sk = sample_point_in_cube(
                        voxel_model_88[
                            i*multiplier:(i+1)*multiplier,
                            j*multiplier:(j+1)*multiplier,
                            k*multiplier:(k+1)*multiplier],
                        voxel_model_temp[i, j, k], halfie)
                    sample_points[batch_size_counter, 0] = si+i*multiplier
                    sample_points[batch_size_counter, 1] = sj+j*multiplier
                    sample_points[batch_size_counter, 2] = sk+k*multiplier
                    sample_values[batch_size_counter, 0] = \
                        voxel_model_temp[i, j, k]
                    voxel_model_temp_flag[i, j, k] = 1
                    batch_size_counter += 1

            sample_points_16 = sample_points
            sample_values_16 = sample_values

            q.put([
                name_list[idx][0],
                exceed_16_flag,
                sample_points_16,
                sample_values_16,
                sample_voxels])


def sample_points_values(voxel_model, res):
    assert int(res / 2) == res / 2
    dim_voxel = int(res / 2)
    voxel_model_temp = np.zeros(
        [dim_voxel, dim_voxel, dim_voxel], np.uint8)
    multiplier = int(res/dim_voxel)
    halfie = int(multiplier/2)

    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_temp[i, j, k] = np.max(
                    voxel_model[
                        i*multiplier:(i+1)*multiplier,
                        j*multiplier:(j+1)*multiplier,
                        k*multiplier:(k+1)*multiplier])

    batch_size = dim_voxel**3
    sample_points = np.zeros([batch_size, 3], np.uint8)
    sample_values = np.zeros([batch_size, 1], np.uint8)
    batch_size_counter = 0
    voxel_model_temp_flag = np.zeros(
        [dim_voxel, dim_voxel, dim_voxel], np.uint8)
    temp_range = list(range(1, dim_voxel-1, 4)) + \
        list(range(2, dim_voxel-1, 4)) + \
        list(range(3, dim_voxel-1, 4)) + \
        list(range(4, dim_voxel-1, 4))
    for j in temp_range:
        if (batch_size_counter >= batch_size):
            break
        for i in temp_range:
            if (batch_size_counter >= batch_size):
                break
            for k in temp_range:
                if (batch_size_counter >= batch_size):
                    break
                if (np.max(
                    voxel_model_temp[i-1:i+2, j-1:j+2, k-1:k+2]) !=
                        np.min(
                        voxel_model_temp[i-1:i+2, j-1:j+2, k-1:k+2])):
                    si, sj, sk = sample_point_in_cube(
                        voxel_model[
                            i*multiplier:(i+1)*multiplier,
                            j*multiplier:(j+1)*multiplier,
                            k*multiplier:(k+1)*multiplier],
                        voxel_model_temp[i, j, k], halfie)
                    sample_points[batch_size_counter, 0] = \
                        si+i*multiplier
                    sample_points[batch_size_counter, 1] = \
                        sj+j*multiplier
                    sample_points[batch_size_counter, 2] = \
                        sk+k*multiplier
                    sample_values[batch_size_counter, 0] = \
                        voxel_model_temp[i, j, k]
                    voxel_model_temp_flag[i, j, k] = 1
                    batch_size_counter += 1
    if (batch_size_counter >= batch_size):
        print("16-- batch_size exceeded!")
    else:
        # fill other slots with random points
        while (batch_size_counter < batch_size):
            while True:
                i = random.randint(0, dim_voxel-1)
                j = random.randint(0, dim_voxel-1)
                k = random.randint(0, dim_voxel-1)
                if voxel_model_temp_flag[i, j, k] != 1:
                    break
            si, sj, sk = sample_point_in_cube(
                voxel_model[
                    i*multiplier:(i+1)*multiplier,
                    j*multiplier:(j+1)*multiplier,
                    k*multiplier:(k+1)*multiplier],
                voxel_model_temp[i, j, k], halfie)
            sample_points[batch_size_counter, 0] = si+i*multiplier
            sample_points[batch_size_counter, 1] = sj+j*multiplier
            sample_points[batch_size_counter, 2] = sk+k*multiplier
            sample_values[batch_size_counter, 0] = \
                voxel_model_temp[i, j, k]
            voxel_model_temp_flag[i, j, k] = 1
            batch_size_counter += 1
    return sample_points, sample_values



def list_image(root, exts):
    image_list = []
    cat = {}
    for path, subdirs, files in os.walk(root):
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                if path not in cat:
                    cat[path] = len(cat)
                image_list.append((os.path.relpath(fpath, root), cat[path]))
    return image_list




if __name__ == '__main__':

    id = "2091"

    # num_items = 2703
    # num_items = 2541
    # num_items = 3
    num_items = 20000
    # exit(0)

    # adj can be either '' or 'augmented_' or 'tmp_' or '256_'
    adj = str(orig_size) + '_'

    obj_name = 'chars'
    if not os.path.exists('../' + str(comp_size) + '_aug_data/'):
        os.mkdir('../' + str(comp_size) + '_aug_data/')

    output_dir = '../' + str(comp_size) + '_aug_data/' + id

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    compress = True
    original_dim = orig_size
    if compress:
        compressed_dim = comp_size
    else:
        compressed_dim = original_dim

    # name of output file
    # hdf5_path = output_dir+'/'+adj+obj_name+'_vox'+str(compressed_dim)+'.hdf5'
    hdf5_path = output_dir+'/'+obj_name+'_vox'+str(compressed_dim)+'.hdf5'

    # record statistics
    fstatistics = open(output_dir+'/statistics.txt', 'w', newline='')

    exceed_16 = 0

    # map processes
    q = Queue()
    # path = '../../raw_data/' + str(orig_size) + '_' + 'vox'
    # path = '../../aug_raw_data/'+id+'/augmented_' + str(orig_size) + '_' + 'vox'
    path = '/home/xzhan2/data/aug_raw_data/'+id+'/augmented_' + str(orig_size) + '_' + 'vox'

    ################# added ###############$
    
    vox_list = list_image(path, ['.binvox'])
    # print(vox_list)

    name_list = []
    for i in range(len(vox_list)):
        imagine=vox_list[i][0]
        name_list.append(imagine[0:-7])
    name_list = misc.sorted_alphanumeric(name_list)
    name_num = len(name_list)

    # print(name_list)

    num_of_process = 16
    list_of_list_of_names = []
    for i in range(num_of_process):
        list_of_names = []
        for j in range(i,name_num,num_of_process):
            list_of_names.append([j, path+'/'+name_list[j]+".binvox"])
        list_of_list_of_names.append(list_of_names)

    # workers = [Process(target=get_points_from_vox, args=(q, path, compress))]

    workers = [
        Process(target=get_points_from_vox_multi_thread,
        args=(q, path, compress, list_of_names)) for list_of_names in list_of_list_of_names]

    for p in workers:
        p.start()

    # reduce process
    hdf5_file = h5py.File(hdf5_path,  'w')
    hdf5_file.create_dataset(
        "voxels",
        [num_items, compressed_dim, compressed_dim, compressed_dim, 1],
        np.uint8)
    out_shape_points = [num_items, batch_size_compressed, 3] if compress else [
        num_items, original_dim**3, 3]
    out_shape_values = [num_items, batch_size_compressed, 1] if compress else [
        num_items, original_dim**3, 1]
    hdf5_file.create_dataset(
        "points_"+str(compressed_dim),
        out_shape_points, np.uint8)
    hdf5_file.create_dataset(
        "values_"+str(compressed_dim),
        out_shape_values, np.uint8)

    while True:
        item_flag = True
        try:
            if compress:
                idx, exceed_16_flag, sample_points_16,\
                    sample_values_16, sample_voxels = q.get(True, 1.0)
            else:
                # -------- DEFINE DESIRED VARIABLE NAMES --------
                idx, orig_points_88, orig_values_88,\
                    orig_voxels = q.get(True, 1.0)
        except queue.Empty:
            item_flag = False

        if item_flag:
            # process result
            exceed_16 = exceed_16 + exceed_16_flag if compress else 0
            hdf5_file["points_"+str(compressed_dim)][idx, :, :] =\
                sample_points_16 if compress else orig_points_88
            hdf5_file["values_"+str(compressed_dim)][idx, :, :] =\
                sample_values_16 if compress else orig_values_88
            hdf5_file["voxels"][idx, :, :, :, :] = \
                sample_voxels if compress else orig_voxels

        allExited = True
        for p in workers:
            if p.exitcode is None:
                allExited = False
                break
        if allExited and q.empty():
            break

    fstatistics.write("total: "+str(num_items)+"\n")
    if compress:
        fstatistics.write("exceed_16: "+str(exceed_16)+"\n")
        fstatistics.write("exceed_16_ratio: "+str(
            float(exceed_16)/num_items)+"\n")

    fstatistics.close()
    hdf5_file.close()
    print("finished")
