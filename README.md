# Project DAT550 

This is the code for the project CLEF2022-CheckThat! Task 1a and 1b. 


Setup:
1. Install the requirements.txt
2. Download the glove embedding from https://nlp.stanford.edu/data/glove.6B.zip, unzip this in the working directory. 

To run the model you have to give the correct arguments. The arguments are dependent on the model. 

For a LSTM run you must spesify: 
  1. subtask(a or b)
  2. learning rate
  3. dropout
  4. hidden dimentions
  5. number of epochs
  6. patient, for the early stopping
  
 For CNN run you must spesify: 
  1. subtask(a or b)
  2. learning rate
  3. dropout
  4. filter sizes
  5. number of epochs
  6. hidden dimentions
  7. patience for the early stopping
  
  the main.py will run with the LSTM as a default model type and all the parameaters have the default values. 
  
  To reproduce the best result for each model on each subtask run following commands: 
  
    For best LSTM on subtask a: 
      - python3 main.py --model_type=LSTM --subtask=a --lr=0.000501455185821353 --dropout=0.073000865154092 --hidden_dim=98 --num_epochs=34 --patience=6
    For best LSTM on subtask b: 
      - python3 main.py --model_type=LSTM --subtask=b --lr=0.000877533971724522 --dropout=0.207934349028203 --hidden_dim=74 --num_epochs=14 --patience=6
    
    For the best CNN on subtask a:
      - python3 main.py --model_type=CNN --subtask=a --lr=0.000257620282464702 --dropout=0.0360844288234835 --filter_sizes=[2] --num_filters=130         
      --num_epochs=15 --hidden_dim=29 --patience=5
      
    For the best CNN on subtask b:
      - python3 main.py --model_type=CNN --subtask=b --lr=0.000569280061594414 --dropout=0.180940543268585 --filter_sizes=[7, 3, 5, 2, 2, 3]  
      --num_filters=72 --num_epochs=13 --hidden_dim=54 --patience=7
    
   


