import numpy as np
import pandas as pd 

from jellyfish import(
    jaro_winkler_similarity,
    jaro_similarity,
    match_rating_comparison
)

def group_similarity(df, column_a, column_b):
    itens_a = df[[column_a]].rename(columns={f'{column_a}':'itens'})
    itens_b = df[[column_b]].rename(columns={f'{column_b}':'itens'})

    df = pd.concat([itens_a, itens_b]).drop_duplicates()
    
    df['predito'] = [1 for i in range(df.shape[0])]
    
    return df

def regex_similarity(word, regex, itens):
    return [i for i in itens for match in regex.finditer(i) if word == match.group(1)]

def creat_class(value_similar, bins, labels):
   return pd.cut(value_similar, bins = bins, labels = labels).astype('string')

def remove_stopwords(list_strings, stopwords):
    output = [t for t in list_strings.split(' ') if t not in stopwords]
    return " ".join(output)

def jaccard_similarity(f1, f2):
    return len(set(f1).intersection(set(f2)))/len(set(f1).union(set(f2)))

def similarity_matrix(list_strings, metric):  
    size = list_strings.shape[0]
    
    matrix = np.empty((list_strings.shape[0], list_strings.shape[0]))
    matrix[:] = np.NaN
    
    if metric == 'jaro_winkler':
        for x in range(size):
            for y in range(x, size):
                matrix[x][y] = jaro_winkler_similarity(list_strings[x], list_strings[y])
    elif metric == 'jaro':
        for x in range(size):
            for y in range(x, size):
                matrix[x][y] = jaro_similarity(list_strings[x], list_strings[y])   
    elif metric == 'jaccard':
        for x in range(size):
            for y in range(x, size):
                matrix[x][y] = jaccard_similarity(list_strings[x], list_strings[y])
    elif metric == 'match_ration':    
        for x in range(size):
            for y in range(x, size):
                matrix[x][y] = match_rating_comparison(list_strings[x], list_strings[y])
    else:
        print('Erro no parametro de metrica')
                
    return matrix