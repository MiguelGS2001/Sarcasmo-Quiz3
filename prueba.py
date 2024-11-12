import unittest
from pipeline import *

class TestEnsayos(unittest.TestCase):
    def test_pipe(self):
        mejor_modelo, mejor_acc_train, mejor_acc, resultados = final()
        resultados.to_csv("resultados.csv")
        self.assertLessEqual(np.abs(mejor_acc_train - mejor_acc), 10,
                                 print("No presenta Underfitting ni Overfitting"))
        
if __name__ == "__main__":
    unittest.main()