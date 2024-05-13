#pragma once

#include <iostream>
#include <vector>

std::vector<std::pair<int, float>> calc_map(const std::string ground_truth_path, 
    const std::string detection_results_path);