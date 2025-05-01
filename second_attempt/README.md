# final_project_460j
A real time ASL to text translator autocorrected with an LLM

# usage:
1. create conda env: 'conda create --name <env_name>
2. activate it: 'conda activate <env_name>
3. install pip: 'conda install pip'
4. get all requirements: 'pip install requirements.txt'
5. should be all up to date

# Using preprocessing script for training CNN model:
1. Download the Kaggle ASL dataset and rename the folder "asl_alphabet_train" to "data" and move it to the same folder as the juypter notebook.
2. After running the script in the notebook, a folder called "train" will be made with all the preprocessed images in each class to train model.

# note:
requirements.txt is created using pipreqs, when adding import statements, make sure to cd a level above, and run the following command: 'pipreqs /path/to/your/project --force'. This will update the requirements.txt file instead of creating a new one.

# How to run the ASL translator
First, create the data set of the landmarks by changing the DATA_DIR variable at the top of the "create_dataset_new.py" python file with the directory of where you have the training data from the github folder. The directory should lead to where all folderes with all the letters are. then run the python file to create the landmark data.

After this finishes running, you will get a .pickle file (it might be titled data.pickle, or data_depth.pickle, or something like that). Make sure that in the "train_classifier_new.py", the "data_dict" variable at the top of this python file has the directory to this .pickle file. After this, run "train_classifier_new.py", this shouldn't take too long.

After this, you will get another .pickle file, (will be titled model.pickle, or model_depth.pickle, or something similar). In the inference_classifier_new.py file, make sure the "model_dict" variable has the file path of the model.pickle file. After this, run the inference_classifier_new.py fiel to test the final product.