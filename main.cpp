#include <iostream>
#include "mAP.h"

int main()
{
    std::string ground_truth_path = "./input/ground-truth";
    std::string detection_results_path = "./input/detection-results";
    std::vector<std::pair<int, float>> map = calc_map(ground_truth_path, detection_results_path);
    for (auto& pair : map)
    {
        std::cout << "class: " << pair.first << " mAP: " << pair.second << std::endl;
    }
    return 0;
}
