#####################################################################################################################################
#                                                                                                                                   #
#                                                          PREPROCESSING MODULE                                                     #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import os
import json
from shutil import copyfile


def read_Captions(filepath):
    '''
        Reading captions from the filepath specified
        Returns a dictionary in a format {imagefilename : [<captions1>, <captions2>, <caption3>...]}
    '''
    caption_dict = {}
    with open(filepath) as f:
        for line in f:
            split_line = line.split(sep = '\t', maxsplit = 1)
            caption = split_line[1][: -1]
            image_id = split_line[0].split(sep = '#')[0]
            if image_id not in caption_dict:
                caption_dict[image_id] = [caption]
            else:
                caption_dict[image_id].append(caption)
            
        return caption_dict


def load_ids(filepath):
    '''
        Function for returning a list of image names from the file passed in args filepath
    '''
    ids = []
    with open(filepath) as f:
        for image_id in f:
            if image_id not in ids:
                ids.append(image_id[:-1])
                
    return ids


def CopyFiles(output_dir, input_dir, ids):
    '''
        Copies files specified by ids from input_dir to output_dir
    '''
    output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for curid in ids:
        input_file = os.path.join(input_dir, curid)
        output_file = os.path.join(output_dir, curid)
        
        copyfile(input_file, output_file)


def writeCaptions(output_dir, caption_dict, ids):
    '''
        Write captions of keys given by ids from caption_dict to captions.txt file in output_dir
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path_file = os.path.join(output_dir, 'captions.txt')
    captions = []
    for curid in ids:
        caption = {curid : caption_dict[curid]}
        
        captions.append(json.dumps(caption))
    
    with open(output_path_file, 'w') as f:
        f.write('\n'.join(captions))

def load_Captions(caption_path):
    '''
        Loading captions from the file at caption path
    '''
    caption_file = os.path.join(caption_path, 'captions.txt')
    captions_dict = {}
    with open(caption_file) as f:
        for line in f:
            cur_line = json.loads(line)
            for k, v in cur_line.items():
                captions_dict[k] = v
    return captions_dict
    
if __name__ == '__main__':
    dir_name_text = '../Data/Flickr8k_text'
    dir_name_images = '../Data/Flickr8k_images'

    filename_token = 'Flickr8k.token.txt'
    filename_train = 'Flickr_8k.trainImages.txt'
    filename_dev = 'Flickr_8k.devImages.txt'
    filename_test = 'Flickr_8k.testImages.txt'
    
    filepath_token = os.path.join(dir_name_text, filename_token)

    # make Caption - Dictionary
    captions_dict = read_Captions(filepath_token)

    # Read train, dev, test image names
    train_ids = load_ids(os.path.join(dir_name_text, filename_train))
    dev_ids = load_ids(os.path.join(dir_name_text, filename_dev))
    test_ids = load_ids(os.path.join(dir_name_text, filename_test))

    # Copy files of train, dev and test in their respective folders
    CopyFiles('../Processed Data/train', dir_name_images, train_ids)
    CopyFiles('../Processed Data/dev', dir_name_images, dev_ids)
    CopyFiles('../Processed Data/test', dir_name_images, test_ids)

    # making Caption files for test, train and dev in the respective folders
    writeCaptions('../Processed Data/train', captions_dict, train_ids)
    writeCaptions('../Processed Data/dev', captions_dict, dev_ids)
    writeCaptions('../Processed Data/test', captions_dict, test_ids)