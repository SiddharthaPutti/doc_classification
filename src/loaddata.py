import os 
import random 
import pickle

def data_reader(path):
    data=[]
    topic_names = os.listdir(path)
    print('loading data...')
    for label, name in enumerate(topic_names):
        data_path = os.path.join(path, name)
        file_names = os.listdir(data_path)
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            with open(file_path, 'r') as f:
                content = f.read()
                data.append([content.replace("\n","").replace("\'",""), label])
                random.shuffle(data)

    with open('loaded_data.pkl', 'wb') as pickel_data:
        pickle.dump(data, pickel_data)

    return data 

def load_data(path, reload= False):
    data_file = 'loaded_data.pkl'
    src_path = "C:/Users/sid31/Downloads/main/NLP/document_classification"
    src_files = os.listdir(src_path)
    if data_file in src_files:
        if reload: 
            data = data_reader(path)
            return data 
        else: 
            print("loading from pickel file...")
            with open('loaded_data.pkl', 'rb') as loaded_data:
                data = pickle.load(loaded_data)
            return data 

    else: 
        data = data_reader(path)
        return data 
    
# if __name__ == "__main__":
#     data = load_data("C:/Users/sid31/Downloads/main/NLP/document_classification/data/data_bbc", reload = True)
#     print(len(data))
    

    

    