import os, shutil

from os import name, system
if name == 'posix':
    system("clear")
    cwd = os.getcwd()
    # Only properly developed for windows
if name == 'nt':
    system('cls')
    cwd = os.getcwd()
    data_dir = 'C:\data\deep_learning\Ch5'    
  
# Make the appropriate directories 
mod_data_dir = os.path.join(data_dir, 'cats_dogs_mod')

train_dir = os.path.join(mod_data_dir, 'train')
validation_dir = os.path.join(mod_data_dir, 'validation')
test_dir = os.path.join(mod_data_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# Iterative 
dir_lst = [ mod_data_dir, 
            train_dir, 
            validation_dir,
            test_dir, 
            train_cats_dir,
            train_dogs_dir,
            validation_cats_dir, 
            validation_dogs_dir,
            test_cats_dir,
            test_dogs_dir]


for directory in dir_lst:
    if not os.path.exists(directory): os.mkdir(directory)

mod_dir_lst = [ train_cats_dir, validation_cats_dir, test_cats_dir,
                train_dogs_dir, validation_dogs_dir, test_dogs_dir]

fnames = [  [f'cat.{i}.jpg' for i in range(1000)], 
            [f'cat.{i}.jpg' for i in range(1000, 1500)],
            [f'cat.{i}.jpg' for i in range(1500, 2000)],
            [f'dog.{i}.jpg' for i in range(1000)], 
            [f'dog.{i}.jpg' for i in range(1000, 1500)],
            [f'dog.{i}.jpg' for i in range(1500, 2000)]]

for idx, fname in enumerate(fnames):
    # if os.list
    for f_nm in fname:
        src = os.path.join(data_dir, f_nm)
        dst = os.path.join(mod_dir_lst[idx], f_nm)
        shutil.copy(src, dst)
    print(f'Total images in {mod_dir_lst[idx]} = {len(os.listdir(mod_dir_lst[idx]))}')

print(f'train_dir = {train_cats_dir}')
print(f'validation_dir= {validation_cats_dir}')
