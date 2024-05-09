import json

def guardar_history(history):
    history_dict = history.history
    with open('history_tb_model_2.json', 'w') as f:
        json.dump(history_dict, f)  
        
def cargar_history():
    with open('history.json', 'r') as f:
        return json.load(f)
        