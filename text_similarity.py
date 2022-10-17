import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
from jellyfish import(
    jaro_winkler_similarity,
    jaro_similarity,
    match_rating_comparison
)

class TextSimilarity:      
    
    def __init__(self):
        self.stopwords = [i.upper() for i in stopwords.words('portuguese')] 

    def regex_prefix(self, word, regex, list_str):
        return [i for i in list_str for match in regex.finditer(i) if word == match.group(1)]
        
    def remove_stopwords(self, list_str):
        output = [t for t in list_str.split(' ') if t not in self.stopwords]
        return " ".join(output)

    
    def creat_class(self, value_similar, bins, labels):
        return pd.cut(value_similar, bins = bins, labels = labels).astype('string')
    

    def group_similarity(self, df, column_a, column_b):
        itens_a = df[[column_a]].rename(columns={f'{column_a}':'itens'})
        itens_b = df[[column_b]].rename(columns={f'{column_b}':'itens'})

        df = pd.concat([itens_a, itens_b]).drop_duplicates()
        df['predito'] = [1 for i in range(df.shape[0])]
        return df
    

    def jaccard_similarity(self, f1, f2):
        return len(set(f1).intersection(set(f2)))/len(set(f1).union(set(f2)))


    def similarity_matrix(self, list_str, metric):  
        size = list_str.shape[0]
        
        matrix = np.empty((list_str.shape[0], list_str.shape[0]))
        matrix[:] = np.NaN
        
        if metric == 'jaro_winkler':
            for x in range(size):
                for y in range(x, size):
                    matrix[x][y] = jaro_winkler_similarity(list_str[x], list_str[y])
        elif metric == 'jaro':
            for x in range(size):
                for y in range(x, size):
                    matrix[x][y] = jaro_similarity(list_str[x], list_str[y])   
        elif metric == 'jaccard':
            for x in range(size):
                for y in range(x, size):
                    matrix[x][y] = self.jaccard_similarity(list_str[x], list_str[y])
        elif metric == 'match_ration':    
            for x in range(size):
                for y in range(x, size):
                    matrix[x][y] = match_rating_comparison(list_str[x], list_str[y])
        else:
            print('Erro de Parametro')
                                
        return matrix