#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

namespace fs = std::filesystem;

// Bitboard Tables
uint16_t ROW_LEFT_TABLE[65536];
uint16_t ROW_RIGHT_TABLE[65536];
uint32_t SCORE_TABLE[65536];
uint16_t REVERSE_ROW_TABLE[65536];

void init_tables() {
  for (int row = 0; row < 65536; ++row) {
    int t[4] = {(row >> 12) & 0xF, (row >> 8) & 0xF, (row >> 4) & 0xF,
                row & 0xF};

    int rev = (t[3] << 12) | (t[2] << 8) | (t[1] << 4) | t[0];
    REVERSE_ROW_TABLE[row] = rev;

    int new_line[4] = {0, 0, 0, 0};
    int non_empty[4];
    int ne_count = 0;
    for (int i = 0; i < 4; ++i) {
      if (t[i] != 0)
        non_empty[ne_count++] = t[i];
    }

    int score = 0;
    int idx = 0;
    for (int i = 0; i < ne_count;) {
      if (i + 1 < ne_count && non_empty[i] == non_empty[i + 1] &&
          non_empty[i] < 15) {
        int val = non_empty[i] + 1;
        new_line[idx++] = val;
        score += (1 << val);
        i += 2;
      } else {
        new_line[idx++] = non_empty[i];
        i += 1;
      }
    }

    uint16_t left_val = (new_line[0] << 12) | (new_line[1] << 8) |
                        (new_line[2] << 4) | new_line[3];
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
  if (action == 0)
    return move_up(board);
  if (action == 1)
    return move_right(board);
  if (action == 2)
    return move_down(board);
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

std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
std::uniform_real_distribution<float> dist(0.0, 1.0);

uint64_t insert_random_tile(uint64_t board) {
  auto empties = get_empty_positions(board);
  if (empties.empty())
    return board;
  int idx = std::uniform_int_distribution<int>(0, empties.size() - 1)(rng);
  int pos = empties[idx];
  uint64_t val = (dist(rng) < 0.9f) ? 1 : 2;
  return board | (val << (pos * 4));
}

bool is_terminal(uint64_t board) {
  if (get_empty_positions(board).size() > 0)
    return false;
  if (move_left(board).first != board)
    return false;
  if (move_right(board).first != board)
    return false;
  if (move_up(board).first != board)
    return false;
  if (move_down(board).first != board)
    return false;
  return true;
}

int reflect_h(int p) { return (p / 4) * 4 + (3 - (p % 4)); }
int reflect_v(int p) { return (3 - (p / 4)) * 4 + (p % 4); }
int reflect_d(int p) { return (p % 4) * 4 + (p / 4); }

// Cœur de l'Intelligence Artificielle (Réseau d'Évaluation N-Tuples)
// Version inférence allégée de 4.29 Go, rétrocompatible avec l'Ultimate TC
class NTupleNetwork {
public:
  std::vector<std::vector<std::vector<int>>> extract_shifts;
  float *LUTS[4];

  NTupleNetwork() {
    std::vector<std::vector<int>> base_shapes = {{0, 1, 2, 3, 4, 5, 6},
                                                 {0, 1, 2, 4, 5, 6, 8},
                                                 {0, 1, 2, 3, 5, 6, 9},
                                                 {0, 1, 2, 4, 5, 8, 9}};

    for (int i = 0; i < 4; ++i) {
      try {
        LUTS[i] = new float[268435456];
      } catch (std::bad_alloc &ba) {
        std::cerr << "OUT OF MEMORY: FATAL ERROR. Impossible d'allouer 4 GB "
                     "pour les 7-Tuples."
                  << std::endl;
        exit(1);
      }
      std::fill_n(LUTS[i], 268435456, 0.0f);

      std::vector<std::vector<int>> isos;
      auto s1 = base_shapes[i];
      auto add_if_unique = [&](const std::vector<int> &s) {
        auto s_sorted = s;
        std::sort(s_sorted.begin(), s_sorted.end());
        for (auto &existing : isos) {
          auto e_sorted = existing;
          std::sort(e_sorted.begin(), e_sorted.end());
          if (s_sorted == e_sorted)
            return;
        }
        isos.push_back(s);
      };

      auto apply = [](const std::vector<int> &s, auto func) {
        std::vector<int> res;
        for (int x : s)
          res.push_back(func(x));
        return res;
      };

      auto h = [&](const std::vector<int> &s) { return apply(s, reflect_h); };
      auto v = [&](const std::vector<int> &s) { return apply(s, reflect_v); };
      auto d = [&](const std::vector<int> &s) { return apply(s, reflect_d); };

      add_if_unique(s1);
      add_if_unique(h(s1));
      add_if_unique(v(s1));
      add_if_unique(v(h(s1)));
      auto s5 = d(s1);
      add_if_unique(s5);
      add_if_unique(h(s5));
      add_if_unique(v(s5));
      add_if_unique(v(h(s5)));

      std::vector<std::vector<int>> shifts;
      for (auto &shape : isos) {
        shifts.push_back(apply(shape, [](int p) { return p * 4; }));
      }
      extract_shifts.push_back(shifts);
    }
  }

  ~NTupleNetwork() {
    for (int i = 0; i < 4; ++i)
      delete[] LUTS[i];
  }

  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::cerr << "Cannot open " << path << std::endl;
      exit(1);
    }
    for (int i = 0; i < 4; ++i) {
      in.read(reinterpret_cast<char *>(LUTS[i]), 268435456 * sizeof(float));
    }
    in.close();
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
        val += LUTS[i][idx];
      }
    }
    return val;
  }
};

// Transposition Table (Cache 32 Mo) pour accélérer x10 l'arbre de recherche
struct TTEntry {
  uint64_t key;
  float value;
  int depth;
};

const int TT_MASK = 0xFFFFFF; // 16.7 M entries
TTEntry* tt_max = new TTEntry[TT_MASK + 1]();
TTEntry* tt_chance = new TTEntry[TT_MASK + 1]();

// Arbre de recherche stochastique modélisant le hasard absolu de 2048 (Expectimax)
float expectimax_chance(uint64_t afterstate, int depth, NTupleNetwork &model);

float expectimax_max(uint64_t board, int depth, NTupleNetwork &model) {
  if (depth == 0) return 0;
  
  int hash = (board ^ (board >> 24) ^ (board >> 48)) & TT_MASK;
  if (tt_max[hash].key == board && tt_max[hash].depth == depth) {
    return tt_max[hash].value;
  }

  float best_v = -1e9;
  for (int a = 0; a < 4; ++a) {
    auto p = move(a, board);
    if (p.first == board) continue;
    float v = p.second + expectimax_chance(p.first, depth - 1, model);
    if (v > best_v) best_v = v;
  }
  
  float res = (best_v == -1e9) ? 0 : best_v;
  tt_max[hash] = {board, res, depth};
  return res;
}

float expectimax_chance(uint64_t afterstate, int depth, NTupleNetwork &model) {
  if (depth == 0) return model.evaluate(afterstate);

  int hash = (afterstate ^ (afterstate >> 24) ^ (afterstate >> 48)) & TT_MASK;
  if (tt_chance[hash].key == afterstate && tt_chance[hash].depth == depth) {
    return tt_chance[hash].value;
  }

  auto empties = get_empty_positions(afterstate);
  if (empties.empty()) return 0;

  float expected_value = 0;
  for (int pos : empties) {
    uint64_t b2 = afterstate | (1ULL << (pos * 4));
    uint64_t b4 = afterstate | (2ULL << (pos * 4));
    expected_value += 0.9f * expectimax_max(b2, depth, model);
    expected_value += 0.1f * expectimax_max(b4, depth, model);
  }
  
  float res = expected_value / empties.size();
  tt_chance[hash] = {afterstate, res, depth};
  return res;
}

int expectimax_best_action(uint64_t board, NTupleNetwork &model, int depth) {
  float best_v = -1e9;
  int best_a = -1;
  for (int a = 0; a < 4; ++a) {
    auto p = move(a, board);
    if (p.first == board) continue;
    float v = p.second + expectimax_chance(p.first, depth - 1, model);
    if (v > best_v) {
      best_v = v;
      best_a = a;
    }
  }
  return best_a;
}

void print_stats(const std::string &name, int num_games, long long total_score,
                 const std::map<int, int> &tiles) {
  std::cout << "=======================================" << std::endl;
  std::cout << " Agent : " << name << std::endl;
  std::cout << "=======================================" << std::endl;
  std::cout << " Parties Jouées : " << num_games << std::endl;
  std::cout << " Score Moyen    : " << (total_score / num_games) << std::endl;
  std::cout << " --- Distribution des Max Tuiles ---" << std::endl;

  for (int p_of_2 = 15; p_of_2 >= 1; --p_of_2) {
    int tile_val = 1 << p_of_2;
    if (tiles.count(tile_val) && tiles.at(tile_val) > 0) {
      float percent = (float)tiles.at(tile_val) / num_games * 100.0f;
      std::cout << " Tuile " << std::setw(5) << tile_val << " : "
                << std::setw(4) << tiles.at(tile_val) << " (" << std::fixed
                << std::setprecision(1) << std::setw(5) << percent << "%)"
                << std::endl;
    }
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  init_tables();
  int NUM_GAMES = 100;

  std::string model_path = "ntuple_model.bin";
  int SEARCH_DEPTH = 2;

  if (argc > 1) {
    fs::path p(argv[1]);
    if (fs::is_directory(p)) p /= "ntuple_model.bin";
    model_path = p.string();
  }
  if (argc > 2) {
    NUM_GAMES = std::atoi(argv[2]);
  }
  if (argc > 3) {
    SEARCH_DEPTH = std::atoi(argv[3]);
  }

  std::cout << "Lancement de l'évaluation sur " << NUM_GAMES << " parties..." << std::endl;
  std::cout << "Profondeur de recherche : " << SEARCH_DEPTH << "-ply (Cache TT activé - 32MB)" << std::endl;

  // 1. EVALUER L'AGENT RANDOM
  long long random_total_score = 0;
  std::map<int, int> random_tiles;

  std::cout << "Evaluation Random... " << std::flush;
  for (int ep = 0; ep < NUM_GAMES; ++ep) {
    uint64_t board = 0;
    board = insert_random_tile(board);
    board = insert_random_tile(board);

    int local_score = 0;
    while (!is_terminal(board)) {
      // Random valid action
      std::vector<int> valid_actions;
      for (int a = 0; a < 4; ++a) {
        if (move(a, board).first != board) {
          valid_actions.push_back(a);
        }
      }
      if (valid_actions.empty())
        break;

      int idx =
          std::uniform_int_distribution<int>(0, valid_actions.size() - 1)(rng);
      int a = valid_actions[idx];

      auto p = move(a, board);
      local_score += p.second;
      board = insert_random_tile(p.first);
    }

    random_total_score += local_score;
    int mx = 0;
    for (int i = 0; i < 16; ++i) {
      int v = (board >> (i * 4)) & 0xF;
      if (v > mx)
        mx = v;
    }
    random_tiles[1 << mx]++;
  }
  std::cout << "Terminé." << std::endl;

  // 2. EVALUER L'AGENT IA
  std::cout << "Chargement du modèle " << model_path << "... " << std::flush;
  NTupleNetwork model;
  model.load(model_path);
  std::cout << "Modèle chargé." << std::endl;

  long long ai_total_score = 0;
  std::map<int, int> ai_tiles;

  std::cout << "Evaluation de l'IA N-Tuple... " << std::flush;
  for (int ep = 0; ep < NUM_GAMES; ++ep) {
    uint64_t board = 0;
    board = insert_random_tile(board);
    board = insert_random_tile(board);

    int local_score = 0;
    while (!is_terminal(board)) {
      int a = expectimax_best_action(board, model, SEARCH_DEPTH);
      if (a == -1) break;

      auto p = move(a, board);
      local_score += p.second;
      board = insert_random_tile(p.first);
    }

    ai_total_score += local_score;
    int mx = 0;
    for (int i = 0; i < 16; ++i) {
      int v = (board >> (i * 4)) & 0xF;
      if (v > mx)
        mx = v;
    }
    ai_tiles[1 << mx]++;
  }
  std::cout << "Terminé.\n" << std::endl;

  // AFFICHER
  print_stats("L'ALÉATOIRE", NUM_GAMES, random_total_score, random_tiles);
  print_stats("IA N-TUPLE (Entraînée)", NUM_GAMES, ai_total_score, ai_tiles);

  return 0;
}
