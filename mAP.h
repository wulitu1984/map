#pragma once

#include <iostream>
#include <vector>

std::vector<std::string> get_files_in_path(const std::string path);
std::vector<std::pair<int, float>> calc_mAP(const std::string ground_truths_path, 
    const std::string detection_results_path);