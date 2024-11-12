import pandas as pd
import numpy as np
import fasttext
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input,Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  accuracy_score, roc_auc_score
from transformers import BertTokenizer, BertModel

def cargar_datos():
    bert_embeddings = pd.read_csv("embeddings_bert.csv", index_col=0)
    fast_embeddings = pd.read_csv("doc_embedding.csv", index_col=0)
    train = pd.read_csv("datos_entrenamiento_ing.csv", index_col=0)
    test = pd.read_csv("datos_test.csv", index_col=0)
    return bert_embeddings, fast_embeddings, train, test

def datos_train(train):
    variables = ["corrections","token_count","unique_token_count","nostop_count","sent_count","ner_count",
                 "comma","question","exclamation","quotation","organization", "caps", "person",
                 "location", "money", "time", "date", "percent", "noun", "adj", "pron",
                 "verb", "cconj", "adv", "det", "propn", "num", "part", "intj"]

    ingieneria = ['exclamation_2', 'noun_2', 'sent_count_2', 'exclamation_x_noun',
                  'nostop_count_x_intj', 'noun_x_cconj']

    features_sin = train.get(variables)
    features_con = train.get(variables+ingieneria)
    return features_con, features_sin

def red(embedding_shape, features_shape):
    x1 = Input(shape=(embedding_shape,), name='Input_Embedding')
    x2 = Input(shape=(features_shape,), name='Input_Features')
    x = Concatenate(name='Concatenar')([x1, x2])
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='elu', name='Capa_Densa_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='elu', name='Capa_Densa_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='elu', name='Capa_Densa_3')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='Output')(x)
    
    model = Model(inputs=[x1, x2], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def index(datos):
    X_provisional = datos.get(["texto","corrected"])
    y_provisional = datos.get(["Sarcasmo"]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_provisional,y_provisional,test_size = 0.10, stratify = y_provisional)
    index_train = X_train.index
    index_test = X_test.index
    return index_train, index_test

def features(feature, index_train, index_test):
    scaler = MinMaxScaler((-1.0,1.0))

    features_train = feature.iloc[index_train,]
    features_train_scaled = pd.DataFrame(scaler.fit_transform(features_train))
    features_train_scaled.columns = features_train.columns
    
    features_test = feature.iloc[index_test,]
    features_test_scaled = pd.DataFrame(scaler.fit_transform(features_test))
    features_test_scaled.columns = features_test.columns

    return features_train_scaled, features_test_scaled

def embeddings(embedding, index_train, index_test):
    embeddings_train = embedding.iloc[index_train,]
    embeddings_test = embedding.iloc[index_test,]
    return embeddings_train, embeddings_test

def modelo(datos, embedding, feature):
    index_train, index_test = index(datos)

    y = datos["Sarcasmo"]
    y_train = y.iloc[index_train,]
    y_test = y.iloc[index_test,]

    features_train_scaled, features_test_scaled = features(feature, index_train, index_test)
    embeddings_train, embeddings_test = embeddings(embedding, index_train, index_test)

    model = red(embeddings_train.shape[1],features_train_scaled.shape[1])
    model.fit(x = [embeddings_train, features_train_scaled], y = y_train, 
              validation_data = ([embeddings_test, features_test_scaled],y_test),
              epochs=100, batch_size=32, verbose=0)

    y_pred = model.predict([embeddings_test, features_test_scaled])
    y_pred = (y_pred>=0.5).astype(int)

    y_pred_train = model.predict([embeddings_train, features_train_scaled])
    y_pred_train = (y_pred>=0.5).astype(int)
    train_accuracy = round(accuracy_score(y_pred_train,y_pred_train),4)

    roc = round(roc_auc_score(y_test,y_pred),4)
    acc = round(accuracy_score(y_test,y_pred),4)
    return model, train_accuracy, roc, acc

def bert(test):
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return cls_embedding.flatten()
    test['bert_embedding'] = test['processed_text'].apply(get_bert_embedding)
    test_embedding = np.stack(test['bert_embedding'].values)
    embedding = pd.DataFrame(test_embedding)
    return embedding

def fast(test):
    ft_model = fasttext.train_unsupervised("Texto_corregido_train.csv",dim=300)
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,),dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.get_word_vector(word))
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    def averaged_word_vectorizer(corpus, model, num_features):
        vocabulary = set(model.words)
        features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
        return np.array(features)
    ftext_feature = averaged_word_vectorizer(corpus=test['tokens'], model=ft_model, num_features=ft_model.dim)
    embedding = pd.DataFrame(ftext_feature)
    return embedding

def datos_test(test):
    categorias = ["corrections","token_count","unique_token_count","nostop_count","sent_count","ner_count",
                  "comma","question","exclamation","quotation","organization", "caps", "person",
                  "location", "money", "time", "date", "percent", "noun", "adj", "pron",
                  "verb", "cconj", "adv", "det", "propn", "num", "part", "intj"]
    test_features = test.get(categorias)
    test_cuadr = pd.DataFrame()
    test_inter = pd.DataFrame()

    variables1 = ['exclamation_2', 'noun_2', 'sent_count_2']
    variables2 = ['exclamation_x_noun', 'nostop_count_x_intj', 'noun_x_cconj']
    for var in categorias:
        test_cuadr[f'{var}_2'] = test_features[var] ** 2
        
    for i in range(len(categorias)):
        for j in range(i + 1, len(categorias)):
            var1 = categorias[i]
            var2 = categorias[j]
        
            new_column_name = f'{var1}_x_{var2}'
            test_inter[new_column_name] = test_features[var1] * test_features[var2]

    test_cuadr1 = test_cuadr.get(variables1)
    test_inter1 = test_inter.get(variables2)
    test_ing = pd.concat([test_cuadr1,test_inter1], axis=1)
    features_ing = pd.concat([test_features,test_ing], axis=1)

    return test_features, features_ing

def valores(y_pred, y_true):
    y_pred1 = (y_pred>=0.5).astype(int)
    roc = round(roc_auc_score(y_true,y_pred1),4)
    acc = round(accuracy_score(y_true,y_pred1),4)
    return roc, acc

def final():
    bert_embeddings, fast_embeddings, train, test = cargar_datos()
    features_con, features_sin = datos_train(train)

    model_bc, train_acc_bc, roc_bc, acc_bc = modelo(train, bert_embeddings, features_con)
    model_bs, train_acc_bs, roc_bs, acc_bs = modelo(train, bert_embeddings, features_sin)

    model_fc, train_acc_fc, roc_fc, acc_fc = modelo(train, fast_embeddings, features_con)
    model_fs, train_acc_fs, roc_fs, acc_fs = modelo(train, fast_embeddings, features_sin)

    metricas = pd.DataFrame({"MODELO":["BERT SIN","BERT CON","FAST SIN","FAST CON"],
                             "ROC":[roc_bs,roc_bc,roc_fs,roc_fc], 
                             "ACCURACY":[acc_bs,acc_bc,acc_fs,acc_fc],
                             "ACC_TRAIN":[train_acc_bs,train_acc_bc,train_acc_fs,train_acc_fc]})
    mejor_modelo = metricas["MODELO"][np.argmax(metricas["ROC"])]
    mejor_acc = metricas["ACCURACY"][np.argmax(metricas["ROC"])]
    mejor_acc_train = metricas["ACC_TRAIN"][np.argmax(metricas["ROC"])]

    embedding_bert = bert(test)
    embedding_fast = fast(test)
    test_features, features_ing = datos_test(test)

    y_pred_bc = model_bc.predict([embedding_bert, features_ing])
    y_pred_bs = model_bs.predict([embedding_bert, test_features])
    y_pred_fs = model_fs.predict([embedding_fast, test_features])
    y_pred_fc = model_fc.predict([embedding_fast, features_ing])

    y_true = test["Sarcasmo"].copy()
    roct_bc, acct_bc = valores(y_pred_bc,y_true)
    roct_bs, acct_bs = valores(y_pred_bs,y_true)
    roct_fc, acct_fc = valores(y_pred_fc,y_true)
    roct_fs, acct_fs = valores(y_pred_fs,y_true)
    resultados = pd.DataFrame({"MODELO":["BERT SIN","BERT CON","FAST SIN","FAST CON"],
                               "ROC":[roct_bs,roct_bc,roct_fs,roct_fc], 
                               "ACCURACY":[acct_bs,acct_bc,acct_fs,acct_fc]})
    
    return mejor_modelo, mejor_acc_train, mejor_acc, resultados