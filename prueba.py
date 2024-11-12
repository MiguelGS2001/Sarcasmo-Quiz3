import unittest
from pipeline import *

class TestEnsayos(unittest.TestCase):
    def test_pipe(self):
        mejor_modelo, mejor_acc_train, mejor_acc, mejor_roc, resultados = final()
        resultados.to_csv("resultados.csv")
        self.assertLessEqual(np.abs(mejor_acc_train - mejor_acc), 10,
                                 print(f"El mejor modelo en el entrenamiento es el {mejor_modelo} con un valor ROC de {mejor_roc}, así mismo este no presenta Underfitting ni Overfitting"))
if __name__ == "__main__":
    unittest.main()