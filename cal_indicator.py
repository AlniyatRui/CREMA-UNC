import math
from itertools import combinations

class ModalitySelector:
    def __init__(self, f_values=None):
        # Define all modalities
        self.modalities = {'D', 'N', 'F'}
        
        # Store function values for all possible combinations
        if f_values is None:
            self.utility_func_values = {
                frozenset([]): 68.98,  
                frozenset(['D']): 70.68,
                frozenset(['N']): 70.80,
                frozenset(['F']): 70.52,
                frozenset(['D', 'N']): 71.08,
                frozenset(['D', 'F']): 71.08,
                frozenset(['N', 'F']): 70.58,
                frozenset(['D', 'N', 'F']): 70.90
            }
        else:
            self.utility_func_values = {frozenset(k): v for k, v in f_values.items()}

    def utility_func(self, subset):
        """Get utility function value for a subset"""
        return self.utility_func_values[frozenset(subset)]

    def get_marginal_gain(self, subset, element):
        """Calculate marginal gain"""
        if element in subset:
            return 0
        new_subset = subset | {element}
        return self.utility_func(new_subset) - self.utility_func(subset)
    
    def __get_normalization_factor(self, modalities):
        """Caculate the normalization factor Z"""
        return self.utility_func(modalities)
    
    def _get_all_subsets(self, modalities):
        """Get all possible subsets of modalities"""
        modalities = list(modalities)
        subsets = []
        for r in range(len(modalities) + 1):
            subsets.extend(combinations(modalities, r))
        return subsets
    
    def compute_shapley_value(self, modality):
        """Compute the scaled Shapley value for individual modality"""
        Z_f = self.__get_normalization_factor(self.modalities) 
        other_modalities = self.modalities - {modality}
        shapley_value = 0
        
        for subset in self._get_all_subsets(other_modalities):
            subset = set(subset)
            marginal = self.get_marginal_gain(subset, modality)
            n = len(self.modalities)
            s = len(subset)
            coef = (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)
            shapley_value += coef * marginal

        return shapley_value / Z_f
    
    def compute_cooperation_score(self, modality_set):
        """Compute cooperation score for a set of modalities"""
        S = set(modality_set) 
        if not S.issubset(self.modalities):
            raise ValueError("Invalid modality set")
        Z_f = self.__get_normalization_factor(S)

        # Calculate contribution of set S as a whole
        phi_S = self.utility_func(S) - self.utility_func(set())

        # Calculate sum of individual contributions
        sum_individual = 0
        for modality in S:
            contribution = self.utility_func({modality}) - self.utility_func(set())
            sum_individual += contribution

        cooperation_score = phi_S - sum_individual
        return phi_S, cooperation_score / Z_f

    


# Usage example
def main():
    # Utility function 1: Use default values
    selector1 = ModalitySelector()
    
    # Utility function 2: Use custom f_values
    custom_f_values_1 = {  # 0.1 acc_7 mosi
        (): 24.38,           # majority class results
        ('T',): 23.32,
        ('A',): 14.52,
        ('V',): 16.47,
        ('T', 'A'): 26.15,
        ('T', 'V'): 24.63,
        ('A', 'V'): 16.04,
        ('V', 'A', 'T'): 26.36
    }
    custom_f_values_2 = {  # 0.2 acc_2 mosi
        (): 56.57,           # majority class results
        ('T',): 71.58,
        ('A',): 54.30,
        ('V',): 55.46,
        ('T', 'A'): 76.33,
        ('T', 'V'): 73.00,
        ('A', 'V'): 52.65,
        ('V', 'A', 'T'): 71.78
    }

    custom_f_values_3 = {  # 0.2 acc_7 mosi
        (): 24.38,           # majority class results
        ('T',): 26.33,
        ('A',): 17.81,
        ('V',): 18.08,
        ('T', 'A'): 32.62,
        ('T', 'V'): 29.53,
        ('A', 'V'): 17.73,
        ('V', 'A', 'T'): 28.22
    }

    custom_f_values_4 = {  # 0.3 acc_2 mosi
        (): 54.79,           # majority class results
        ('T',): 74.91,
        ('A',): 53.47,
        ('V',): 56.01,
        ('T', 'A'): 76.70,
        ('T', 'V'): 75.54,
        ('A', 'V'): 55.02,
        ('V', 'A', 'T'): 76.24
    }

    custom_f_values_5 = {  # 0.3 acc_7 mosi
        (): 24.38,           # majority class results
        ('T',): 30.20,
        ('A',): 16.39,
        ('V',): 20.03,
        ('T', 'A'): 34.08,
        ('T', 'V'): 34.43,
        ('A', 'V'): 19.94,
        ('V', 'A', 'T'): 32.54
    }
    custom_f_values_6 = {  # 0.05 acc_2 mosi
        (): 56.14,           # majority class results
        ('T',): 62.20,
        ('A',): 48.11,
        ('V',): 52.04,
        ('T', 'A'): 68.02,
        ('T', 'V'): 65.72,
        ('A', 'V'): 47.84,
        ('V', 'A', 'T'): 62.36
    }
    custom_f_values_7 = {  # 0.05 acc_7 mosi
        (): 24.38,           # majority class results
        ('T',): 23.29,
        ('A',): 15.40,
        ('V',): 17.64,
        ('T', 'A'): 25.19,
        ('T', 'V'): 23.58,
        ('A', 'V'): 16.82,
        ('V', 'A', 'T'): 23.38
    }
    custom_f_values_8 = {  # 0.4 acc_2 mosi
        (): 54.41,           # majority class results
        ('T',): 73.84,
        ('A',): 56.71,
        ('V',): 56.98,
        ('T', 'A'): 75.02,
        ('T', 'V'): 77.52,
        ('A', 'V'): 55.63,
        ('V', 'A', 'T'): 77.26
    }
    custom_f_values_9 = {  # 0.4 acc_7 mosi
        (): 24.38,           # majority class results
        ('T',): 29.53,
        ('A',): 17.78,
        ('V',): 19.65,
        ('T', 'A'): 33.24,
        ('T', 'V'): 35.83,
        ('A', 'V'): 18.37,
        ('V', 'A', 'T'): 33.99
    }
    custom_f_values_10 = {  # 0.1 acc_2 mosei
        (): 62.61,           # majority class results
        ('T',): 74.79,
        ('A',): 59.40,
        ('V',): 58.48,
        ('T', 'A'): 69.89,
        ('T', 'V'): 69.20,
        ('A', 'V'): 52.84,
        ('V', 'A', 'T'): 69.83
    }
    custom_f_values_11 = {  # 0.1 acc_7 mosei
        (): 41.64,           # majority class results
        ('T',): 45.64,
        ('A',): 41.34,
        ('V',): 40.88,
        ('T', 'A'): 47.07,
        ('T', 'V'): 47.35,
        ('A', 'V'): 36.88,
        ('V', 'A', 'T'): 47.30
    }
    custom_f_values_12 = {  # 0.2 acc_2 mosei
        (): 61.43,           # majority class results
        ('T',): 76.36,
        ('A',): 60.16,
        ('V',): 59.89,
        ('T', 'A'): 70.92,
        ('T', 'V'): 70.35,
        ('A', 'V'): 52.57,
        ('V', 'A', 'T'): 71.07
    }
    custom_f_values_13 = {  # 0.2 acc_7 mosei
        (): 41.64,           # majority class results
        ('T',): 48.06,
        ('A',): 41.36,
        ('V',): 41.39,
        ('T', 'A'): 49.61,
        ('T', 'V'): 49.23,
        ('A', 'V'): 37.35,
        ('V', 'A', 'T'): 48.96
    }
    custom_f_values_14 = {  # 0.3 acc_2 mosei
        (): 61.28,           # majority class results
        ('T',): 76.37,
        ('A',): 60.33,
        ('V',): 59.31,
        ('T', 'A'): 70.47,
        ('T', 'V'): 69.54,
        ('A', 'V'): 53.89,
        ('V', 'A', 'T'): 71.29
    }
    custom_f_values_15 = {  # 0.3 acc_7 mosei
        (): 41.64,           # majority class results
        ('T',): 48.70,
        ('A',): 41.32,
        ('V',): 41.39,
        ('T', 'A'): 49.45,
        ('T', 'V'): 49.90,
        ('A', 'V'): 37.63,
        ('V', 'A', 'T'): 49.02
    }
    custom_f_values_16 = {  # 0.4 acc_2 mosei
        (): 61.94,           # majority class results
        ('T',): 76.28,
        ('A',): 60.08,
        ('V',): 59.36,
        ('T', 'A'): 70.33,
        ('T', 'V'): 70.56,
        ('A', 'V'): 55.19,
        ('V', 'A', 'T'): 71.21
    }
    custom_f_values_17 = {  # 0.4 acc_7 mosei
        (): 41.64,           # majority class results
        ('T',): 48.83,
        ('A',): 41.27,
        ('V',): 40.96,
        ('T', 'A'): 49.45,
        ('T', 'V'): 50.94,
        ('A', 'V'): 38.98,
        ('V', 'A', 'T'): 49.74
    }


    selector2 = ModalitySelector()

    selector = selector2

    # Analyze individual modality importance (Shapley values)
    print("\n1. Individual Modality Importance (Shapley Values):")
    for modality in selector.modalities:
        value = selector.compute_shapley_value(modality)
        print(f"Modality {modality}: {value:.3f}")

    
    # Analyze pair-wise cooperation
    print("\n2. Pair-wise Cooperation Scores:")
    modality_list = list(selector.modalities)
    for i in range(len(modality_list)):
        for j in range(i+1, len(modality_list)):
            pair = {modality_list[i], modality_list[j]}
            score, score_wo_unimodal = selector.compute_cooperation_score(pair)
            print(f"Modalities {pair}: {score:.3f}, without unimodal: {score_wo_unimodal:.3f}")

    # Analyze total cooperation
    # print("\n3. Total Cooperation Score:")
    # total_score = selector.compute_cooperation_score(selector.modalities)
    # print(f"All modalities: {total_score:.3f}")

    # average S+C
    

if __name__ == "__main__":
    main()