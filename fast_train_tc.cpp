#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <atomic>
#include <thread>
#include <omp.h>
#include <cmath>
#include <csignal>

namespace fs = std::filesystem;

// Bitboard Tables
uint16_t ROW_LEFT_TABLE[65536];
uint16_t ROW_RIGHT_TABLE[65536];
uint32_t SCORE_TABLE[65536];
uint16_t REVERSE_ROW_TABLE[65536];

// Fonctions de pre-calcul global pour les Bitboards (O(1) execution time)
void init_tables() {
  for (int row = 0; row < 65536; ++row) {
    int t[4] = {(row >> 12) & 0xF, (row >> 8) & 0xF, (row >> 4) & 0xF, row & 0xF};
    int rev = (t[3] << 12) | (t[2] << 8) | (t[1] << 4) | t[0];
    REVERSE_ROW_TABLE[row] = rev;
    int new_line[4] = {0, 0, 0, 0};
    int non_empty[4];
    int ne_count = 0;
    for (int i = 0; i < 4; ++i) {
      if (t[i] != 0) non_empty[ne_count++] = t[i];
    }
    int score = 0;
    int idx = 0;
    for (int i = 0; i < ne_count;) {
      if (i + 1 < ne_count && non_empty[i] == non_empty[i + 1] && non_empty[i] < 15) {
        int val = non_empty[i] + 1;
        new_line[idx++] = val;
        score += (1 << val);
        i += 2;
      } else {
        new_line[idx++] = non_empty[i];
        i += 1;
      }
    }
    uint16_t left_val = (new_line[0] << 12) | (new_line[1] << 8) | (new_line[2] << 4) | new_line[3];
    ROW_LEFT_TABLE[row] = left_val;
    SCORE_TABLE[row] = score;
  }
  for (int row = 0; row < 65536; ++row) {
    int rev = REVERSE_ROW_TABLE[row];
    int rev_left = ROW_LEFT_TABLE[rev];
    ROW_RIGHT_TABLE[row] = REVERSE_ROW_TABLE[rev_left];
  }
}

inline uint64_t transpose(uint64_t x) {
  uint64_t res = 0;
  for (int i = 0; i < 16; ++i) {
    int r = i / 4;
    int c = i % 4;
    int j = c * 4 + r;
    uint64_t val = (x >> (i * 4)) & 0xF;
    res |= (val << (j * 4));
  }
  return res;
}

inline std::pair<uint64_t, int> move_left(uint64_t board) {
  uint64_t res = 0;
  int score = 0;
  for (int i = 0; i < 4; ++i) {
    int row = (board >> ((3 - i) * 16)) & 0xFFFF;
    res |= (uint64_t)(ROW_LEFT_TABLE[row]) << ((3 - i) * 16);
    score += SCORE_TABLE[row];
  }
  return {res, score};
}

inline std::pair<uint64_t, int> move_right(uint64_t board) {
  uint64_t res = 0;
  int score = 0;
  for (int i = 0; i < 4; ++i) {
    int row = (board >> ((3 - i) * 16)) & 0xFFFF;
    res |= (uint64_t)(ROW_RIGHT_TABLE[row]) << ((3 - i) * 16);
    score += SCORE_TABLE[row];
  }
  return {res, score};
}

inline std::pair<uint64_t, int> move_up(uint64_t board) {
  auto p = move_left(transpose(board));
  return {transpose(p.first), p.second};
}

inline std::pair<uint64_t, int> move_down(uint64_t board) {
  auto p = move_right(transpose(board));
  return {transpose(p.first), p.second};
}

inline std::pair<uint64_t, int> move(int action, uint64_t board) {
  if (action == 0) return move_up(board);
  if (action == 1) return move_right(board);
  if (action == 2) return move_down(board);
  return move_left(board);
}

std::vector<int> get_empty_positions(uint64_t board) {
  std::vector<int> empty;
  empty.reserve(16);
  for (int i = 0; i < 16; ++i) {
    if (((board >> (i * 4)) & 0xF) == 0)
      empty.push_back(i);
  }
  return empty;
}

uint64_t insert_random_tile(uint64_t board) {
  thread_local std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
  thread_local std::uniform_real_distribution<float> dist(0.0, 1.0);

  auto empties = get_empty_positions(board);
  if (empties.empty()) return board;
  int idx = std::uniform_int_distribution<int>(0, empties.size() - 1)(rng);
  int pos = empties[idx];
  uint64_t val = (dist(rng) < 0.9f) ? 1 : 2;
  return board | (val << (pos * 4));
}

bool is_terminal(uint64_t board) {
  if (get_empty_positions(board).size() > 0) return false;
  if (move_left(board).first != board) return false;
  if (move_right(board).first != board) return false;
  if (move_up(board).first != board) return false;
  if (move_down(board).first != board) return false;
  return true;
}

// TC Learning Tuple Structure
// Structure requise pour le Temporal Coherence (Historique des erreurs)
struct TupleEntry {
    float weight;
    float delta_sum;
    float abs_delta_sum;
};

int reflect_h(int p) { return (p / 4) * 4 + (3 - (p % 4)); }
int reflect_v(int p) { return (3 - (p / 4)) * 4 + (p % 4); }
int reflect_d(int p) { return (p % 4) * 4 + (p / 4); }

// Cœur de l'Intelligence Artificielle (Réseau d'Évaluation N-Tuples)
// Construit une matrice massive 4x7-Tuples exploitant 12.87 Go de RAM
class NTupleNetwork {
public:
  std::vector<std::vector<std::vector<int>>> extract_shifts;
  TupleEntry *LUTS[4];

  NTupleNetwork(float init_value = 0.0f) {
    std::vector<std::vector<int>> base_shapes = {
        {0, 1, 2, 3, 4, 5, 6},
        {0, 1, 2, 4, 5, 6, 8},
        {0, 1, 2, 3, 5, 6, 9},
        {0, 1, 2, 4, 5, 8, 9}
    };

    for (int i = 0; i < 4; ++i) {
      try {
          // 4 Tables de 16^7 entrées (3.22 GB par table !)
          LUTS[i] = new TupleEntry[268435456];
      } catch (std::bad_alloc& ba) {
          std::cerr << "OUT OF MEMORY: Impossible d'allouer les 12.87 GB requis par le programme !" << std::endl;
          exit(1);
      }
      
      #pragma omp parallel for
      for (int j = 0; j < 268435456; ++j) {
          LUTS[i][j].weight = init_value;
          LUTS[i][j].delta_sum = 0.0f;
          LUTS[i][j].abs_delta_sum = 0.0f;
      }

      std::vector<std::vector<int>> isos;
      auto s1 = base_shapes[i];
      auto add_if_unique = [&](const std::vector<int> &s) {
        auto s_sorted = s;
        std::sort(s_sorted.begin(), s_sorted.end());
        for (auto &existing : isos) {
          auto e_sorted = existing;
          std::sort(e_sorted.begin(), e_sorted.end());
          if (s_sorted == e_sorted) return;
        }
        isos.push_back(s);
      };

      auto apply = [](const std::vector<int> &s, auto func) {
        std::vector<int> res;
        for (int x : s) res.push_back(func(x));
        return res;
      };

      auto h = [&](const std::vector<int> &s) { return apply(s, reflect_h); };
      auto v = [&](const std::vector<int> &s) { return apply(s, reflect_v); };
      auto d = [&](const std::vector<int> &s) { return apply(s, reflect_d); };

      add_if_unique(s1);    add_if_unique(h(s1));    add_if_unique(v(s1));    add_if_unique(v(h(s1)));
      auto s5 = d(s1);
      add_if_unique(s5);    add_if_unique(h(s5));    add_if_unique(v(s5));    add_if_unique(v(h(s5)));

      std::vector<std::vector<int>> shifts;
      for (auto &shape : isos) {
        shifts.push_back(apply(shape, [](int p) { return p * 4; }));
      }
      extract_shifts.push_back(shifts);
    }
  }

  ~NTupleNetwork() {
    for (int i = 0; i < 4; ++i) delete[] LUTS[i];
  }

  bool load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    for (int i = 0; i < 4; ++i) {
      in.read(reinterpret_cast<char *>(LUTS[i]), 268435456 * sizeof(TupleEntry));
    }
    return true;
  }

  void save_checkpoint(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    for (int i = 0; i < 4; ++i) {
      out.write(reinterpret_cast<char *>(LUTS[i]), 268435456 * sizeof(TupleEntry));
    }
  }

  float evaluate(uint64_t board) {
    float val = 0.0f;
    for (int i = 0; i < 4; ++i) {
      for (auto &shifts : extract_shifts[i]) {
        uint32_t idx = (((board >> shifts[0]) & 0xF)) |
                       (((board >> shifts[1]) & 0xF) << 4) |
                       (((board >> shifts[2]) & 0xF) << 8) |
                       (((board >> shifts[3]) & 0xF) << 12) |
                       (((board >> shifts[4]) & 0xF) << 16) |
                       (((board >> shifts[5]) & 0xF) << 20) |
                       (((board >> shifts[6]) & 0xF) << 24);
        val += LUTS[i][idx].weight;
      }
    }
    return val;
  }

  // Mise à jour de l'IA sans verrous (Hogwild! algorithm)
  // Calcule un Alpha dynamique grace au Temporal Coherence pour eviter la divergence
  void update(uint64_t board, float td_error, float current_alpha) {
    int count = 0;
    for (int i = 0; i < 4; ++i) count += extract_shifts[i].size();
    
    // Le "global alpha adjustment" basé sur la taille de l'erreur
    float base_adj = (current_alpha * td_error) / count;

    for (int i = 0; i < 4; ++i) {
      for (auto &shifts : extract_shifts[i]) {
        uint32_t idx = (((board >> shifts[0]) & 0xF)) |
                       (((board >> shifts[1]) & 0xF) << 4) |
                       (((board >> shifts[2]) & 0xF) << 8) |
                       (((board >> shifts[3]) & 0xF) << 12) |
                       (((board >> shifts[4]) & 0xF) << 16) |
                       (((board >> shifts[5]) & 0xF) << 20) |
                       (((board >> shifts[6]) & 0xF) << 24);
                       
        auto& entry = LUTS[i][idx];
        
        // Temporal Coherence Tracking (Hogwild Lock-Free Updates)
        entry.delta_sum += td_error;
        entry.abs_delta_sum += std::abs(td_error);
        
        float local_alpha = 1.0f;
        if (entry.abs_delta_sum > 1e-5f) {
            local_alpha = std::abs(entry.delta_sum) / entry.abs_delta_sum;
        }

        entry.weight += base_adj * local_alpha;
      }
    }
  }
};

int expectimax_best_action(uint64_t board, NTupleNetwork &model) {
  float best_v = -1e9;
  int best_a = -1;
  for (int a = 0; a < 4; ++a) {
    auto p = move(a, board);
    uint64_t afterstate = p.first;
    if (afterstate == board) continue;

    float val = p.second + model.evaluate(afterstate);
    if (val > best_v) {
      best_v = val;
      best_a = a;
    }
  }
  return best_a;
}

std::atomic<bool> keep_running(true);

void signal_handler(int signum) {
    std::cout << "\n[!] Signal d'interruption (Ctrl+C) recu. Sauvegarde du Run en cours..." << std::endl;
    keep_running = false;
}

// Boucle Principale d'Entraînement Asynchrone
// Lance des dizaines de milliers de parties simultanees via OpenMP (20 coeurs)
int main(int argc, char **argv) {
  omp_set_num_threads(20);
  std::signal(SIGINT, signal_handler);

  init_tables();
  int num_episodes = 100000;
  std::string load_path = "";

  if (argc > 1) {
    num_episodes = std::atoi(argv[1]);
  }
  if (argc > 2) {
    load_path = argv[2];
  }

  auto now = std::time(nullptr);
  char buf[100];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
  std::string run_dir = "runs_tc/run_" + std::string(buf);
  fs::create_directories(run_dir);

  std::string stats_file = run_dir + "/stats.csv";
  std::string model_file = run_dir + "/ntuple_model.bin";
  std::string checkpoint_file = run_dir + "/tc_checkpoint.bin";
  std::ofstream csv(stats_file);
  csv << "Episode,AvgScore,MaxTile,TimeSeconds\n";

  std::cout << "============================================================" << std::endl;
  std::cout << "Starting TEMPORAL COHERENCE C++ training for " << num_episodes << " episodes" << std::endl;
  std::cout << "Architecture : 4x7-Tuples (Allocation RAM: 12.87 GB!!)" << std::endl;
  std::cout << "Threading    : 20 cores (Hogwild! openMP + Adaptive Alpha)" << std::endl;
  std::cout << "Saving       : " << run_dir << " (4.29 GB Compatible Format)" << std::endl;
  std::cout << "============================================================" << std::endl;

  NTupleNetwork model(100000.0f); // Fixed Init, global alpha handled dynamically

  if (!load_path.empty()) {
      std::cout << "Tentative de reprise du checkpoint TC : " << load_path << "..." << std::endl;
      if (model.load_checkpoint(load_path)) {
          std::cout << "SUCCES: Memoire et Historique des gradients (12.8 GB) restaures avec succes." << std::endl;
      } else {
          std::cerr << "ECHEC: Impossible de charger le fichier " << load_path << std::endl;
          return 1;
      }
  }

  std::atomic<long long> total_score(0);
  std::atomic<int> max_overall(0);
  std::atomic<int> episodes_done(0);

  auto start_time = std::chrono::high_resolution_clock::now();
  double last_save_time = 0.0;

  // Initializing global backup buffer so it's not repeatedly allocated 
  // It requires 1 extra GB, bringing total system map to ~13.9 GB perfectly safe.
  float* extraction_buffer = nullptr;
  try {
      extraction_buffer = new float[268435456];
  } catch(...) {
      std::cerr << "OUT OF MEMORY: Can't allocate backup extraction buffer." << std::endl;
      return 1;
  }

  #pragma omp parallel for schedule(dynamic, 10)
  for (int ep = 0; ep < num_episodes; ++ep) {
    if (!keep_running) continue; // Skip iteration gracefully

    float current_alpha = 0.01f;
    if (ep >= num_episodes * 3 / 4) current_alpha = 0.0025f;
    else if (ep >= num_episodes / 2) current_alpha = 0.005f;

    uint64_t board = 0;
    board = insert_random_tile(board);
    board = insert_random_tile(board);
    int local_score = 0;
    int step_count = 0;

    while (!is_terminal(board)) {
      if (++step_count > 1000000) {
        #pragma omp critical
        std::cout << "HANG DETECTED in episode " << ep << std::endl;
        break;
      }

      int a = expectimax_best_action(board, model);
      if (a == -1) break;

      auto p = move(a, board);
      uint64_t afterstate = p.first;
      float r = p.second;
      local_score += p.second;

      uint64_t next_state = insert_random_tile(afterstate);
      float target_value = 0.0f;
      if (!is_terminal(next_state)) {
        int next_a = expectimax_best_action(next_state, model);
        if (next_a != -1) {
          auto next_p = move(next_a, next_state);
          target_value = next_p.second + model.evaluate(next_p.first);
        }
      }

      float current_value = model.evaluate(afterstate);
      float td_error = target_value - current_value;
      model.update(afterstate, td_error, current_alpha);
      board = next_state;
    }

    total_score += local_score;
    int mx = 0;
    for (int i = 0; i < 16; ++i) {
      int v = (board >> (i * 4)) & 0xF;
      if (v > mx) mx = v;
    }
    
    int current_max = max_overall.load();
    while (mx > current_max && !max_overall.compare_exchange_weak(current_max, mx)) {}

    int ed = ++episodes_done;

    if (ed % 1000 == 0 || ed == num_episodes) {
      #pragma omp critical
      {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;

        if (ed % 1000 == 0 || ed == num_episodes) {
            csv << ed << "," << (total_score.load() / ed) << ","
                << (1 << max_overall.load()) << "," << elapsed.count() << "\n";

            std::cout << "\rEp " << std::setw(6) << ed << " / " << num_episodes
                      << " | Avg Score: " << (total_score.load() / ed)
                      << " | Max Tile: " << (1 << max_overall.load())
                      << " | Time: " << std::fixed << std::setprecision(2)
                      << elapsed.count() << "s | RAM: 13GB" << std::flush;

            // Autosave every 120 seconds
            if (elapsed.count() - last_save_time > 120.0) {
              last_save_time = elapsed.count();
              std::ofstream out(model_file, std::ios::binary);
              for (int i = 0; i < 4; ++i) {
                // Extract ONLY the 'weight' to maintain compatibility (4.29GB file format instead of 12.87GB)
                for (int j = 0; j < 268435456; ++j) {
                    extraction_buffer[j] = model.LUTS[i][j].weight;
                }
                out.write(reinterpret_cast<char *>(extraction_buffer), 268435456 * sizeof(float));
              }
              out.close();
              csv.flush();
            }
        }
      }
    }
  }

  std::cout << "\nTraining terminie ou interrompu. Best tile: " << (1 << max_overall.load()) << std::endl;
  
  std::cout << "1. Sauvegarde du Modele compatible 3-ply dans " << model_file << " (Extraction 4.29 GB)..." << std::endl;
  std::ofstream out(model_file, std::ios::binary);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 268435456; ++j) {
        extraction_buffer[j] = model.LUTS[i][j].weight;
    }
    out.write(reinterpret_cast<char *>(extraction_buffer), 268435456 * sizeof(float));
  }
  out.close();
  
  std::cout << "2. Creation du Gros Checkpoint pour reprise TC dans " << checkpoint_file << " (Ecriture brute 12.87 GB)..." << std::endl;
  model.save_checkpoint(checkpoint_file);

  csv.close();
  delete[] extraction_buffer;
  std::cout << "L'integralite des poids a ete sauvee, au Revoir et Bonne re-compilation sur evaluate_ultimate !" << std::endl;
  return 0;
}
