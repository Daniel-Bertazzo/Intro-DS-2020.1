from builtins import range
from builtins import object
import numpy as np


class KNearestNeighbor(object):
    """ Classficador kNN """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Como mostrado em aula, o modelo kNN simplesmente se lembra dos dados
        para, durante o teste, comparar com a instancia de inferencia, e isso
        eh exatamente o que este metodo faz.

        Inputs:
        - X: Array numpy de forma (num_train, D)
        - y: Array numpy de forma (N,), em que y[i] e o label de X[i]
        """
        self.X_train = X
        self.y_train = y
        print("Train format: ", self.X_train.shape)
        print("Train format: ", self.y_train.shape)

    def predict(self, X, k=1):
        """
        Realiza a predicao conforme os k vizinhos mais proximos
        
        Inputs:
        - X: Array de teste, analogo ao de treino.
        - k: O numero de vizinhos utilizados para a classificacao
        - num_loops: Qual implementacao utilizar

        Returns:
        - y: Array numpy com todos labels y[i] de cada instancia X[i].
        """
        # Escreva seu codigo entre estas duas linhas
        pass
        # Escreva seu codigo entre estas duas linhas

    def compute_distances_two_loops(self, X):
        """
        Computa a distancia entre cada ponto de teste em X e cada ponto de
        treino em self.X_train usando loops aninhados para os dados de treino e
        teste.

        Inputs:
        - X: Array numpy (num_test, D) com os dados de teste

        Returns:
        - dists: Array numpy (num_test, num_train) onde a dists[i, j] e a 
        distancia Euclidiana entre o i-esimo ponto de teste e o j-esimo ponto
        de treino.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # Itera sobre o conjunto de testes
        for i in range(num_test):
            # Itera sobre o conjunto de treino
            for j in range(num_train):

                # TODO:
                # Realize a computação da distancia L2 (Euclidiana)
                # entre o i-esimo ponto do teste e o j-esimo ponto
                # do treino e armazene o resultado em dists[i, j]
                # Nao use mais que estes dois loops, nem np.linalg.norm().

                # Escreva seu codigo entre estas duas linhas
                dists[i, j] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train[j, :])))
                # Escreva seu codigo entre estas duas linhas

        return dists

    def predict_labels(self, dists, k=1):
        """
        Dada uma matriz de distancias entre pontos de teste e pontos de treino,
        prediz um label para cada ponto de teste.

        Inputs:
        - dists: Um array numpy de forma (num_test, num_train), onde dists[i, j]
        e a distancia entre o i-esimo elemento do teste e o j-esimo elemento
        do treino.

        Returns:
        - y: Numpy array com o formato (num_test,) com os labels das instancias
        de teste.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        # Loop que itera por cada linha da matriz de distancias (dists)
        for i in range(num_test):

            # Lista na qual serao armazenados os k vizinhos mais proximos da
            # iteracao atual do loop.
            closest_y = []

            # TODO:
            # Use a matriz de distancias passada nos argumentos da funcao
            # para encontrar os 'k' vizinhos mais proximos da i-esima instancia
            # de teste.
            # DICA: Use a funcao numpy.argsort.

            # Escreva seu codigo entre estas duas linhas
            temp = np.argsort(dists[i, :])  # Ve quais sao os indices dos mais proximos (temp.shape = (num_test,) )
            # Insere na lista os k mais proximos
            for j in range(k):
                closest_y.append(temp[j]) # Armazena os indices dos k vizinhos mais proximos
            # Escreva seu codigo entre estas duas linhas ^

            # TODO:
            # Agora que voce possui os k vizinhos, realize uma votacao para
            # descobrir qual dos labels e o mais frequente. Em caso de empate,
            # escolha o menor dos labels.
            # DICA: Pesquise use as funcoes do numpy.

            # Escreva seu codigo entre estas duas linhas
            
            # vetor que armazena a frequencia de cada label
            neighbor_labels_freq = np.zeros(np.unique(self.y_train).shape[0]).astype(np.int)
            for j in closest_y:
                # Incrementa a quantidade de ocorrencias do label do j-esimo vizinho mais proximo
                neighbor_labels_freq[self.y_train[j]] += 1

            y_pred[i] = np.argmax(neighbor_labels_freq)

            # Escreva seu codigo entre estas duas linhas ^
        
        return y_pred

