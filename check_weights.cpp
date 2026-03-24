#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./check_weights <path_to_bin>\n";
        return 1;
    }
    std::string path = argv[1];
    
    std::ifstream in(path, std::ios::binary);
    if (!in) { 
        std::cout << "Erreur: Impossible de lire " << path << "\n"; 
        return 1; 
    }
    
    double sum = 0;
    float mx = -1e9;
    float mn = 1e9;
    long long untouched = 0;
    long long total = 4LL * 268435456LL; // 4 tables, 16^7 entrées
    
    float* buffer = new float[268435456];
    
    std::cout << "Analyse en cours du monstre de 4.29 Go...\n";
    
    for (int i = 0; i < 4; ++i) {
        in.read(reinterpret_cast<char*>(buffer), 268435456 * sizeof(float));
        for (int j = 0; j < 268435456; ++j) {
            float v = buffer[j];
            if (v == 100000.0f) {
                untouched++;
            } else {
                if (v > mx) mx = v;
                if (v < mn) mn = v;
                sum += v;
            }
        }
    }
    
    long long touched = total - untouched;
    
    std::cout << "\n================ RESULTATS DE CONVERGENCE ================\n";
    std::cout << "Poids Totaux         : " << total << " cases mémoire\n";
    std::cout << "Modifiés (Appris)    : " << touched << " (" << std::fixed << std::setprecision(4) << (touched * 100.0 / total) << "%)\n";
    std::cout << "Vierges (100000.0)   : " << untouched << " (" << std::fixed << std::setprecision(4) << (untouched * 100.0 / total) << "%)\n";
    std::cout << "----------------------------------------------------------\n";
    if (touched > 0) {
        std::cout << "Poids Min            : " << mn << "\n";
        std::cout << "Poids Max (Gagnant)  : " << mx << "\n";
        std::cout << "Moyenne des modifiés : " << (sum / touched) << "\n";
    }
    std::cout << "==========================================================\n";
    
    delete[] buffer;
    return 0;
}
