from sklearn.metrics import f1_score 


def get_f1(x, y, f1_type):
  f1 = f1_score(y, x, average = f1_type)
  return f1
    
    
  
  

  
  

