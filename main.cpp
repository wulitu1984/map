#include <iostream>
#include "mAP.h"

int main()
{
    std::string ground_truths_path = "./input/ground-truth";
    std::string detection_results_path = "./input/detection-results";
    std::vector<std::pair<int, float>> map = calc_mAP(ground_truths_path, detection_results_path);
    float map_all = 0;
    int cnt = 0;
    for (auto& pair : map)
    {
        std::cout << "class: " << pair.first << " mAP: " << pair.second << std::endl;
        cnt++;
        map_all+= pair.second;
    }
    std::cout << "total: " << map_all/cnt << std::endl;
    return 0;
}
