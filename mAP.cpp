#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <dirent.h>
#include <algorithm>    
#include <map>
#include "mAP.h"

std::vector<std::string> get_files_in_path(const std::string path)
{
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string file_name = ent->d_name;
            if (file_name != "." && file_name != "..")
            {
                files.push_back(path + "/" + file_name);
            }
        }
        closedir(dir);
    }
    else
    {
        /* could not open directory */
        perror("");
    }

    //sort files by name
    std::sort(files.begin(), files.end());
    return files;
}

struct groud_truth
{
    int cls;
    std::vector<float> bbox;//xyxy
    bool used;
};

struct groud_truth_info
{
    std::map<int, int> gt_counter_per_class;
    std::map<int, int> counter_images_per_class;
};


std::pair<std::vector<groud_truth>, groud_truth_info> get_ground_truth(const std::string ground_truth_path)
{
    std::vector<groud_truth> ground_truths;
    groud_truth_info info;
    std::vector<std::string> files = get_files_in_path(ground_truth_path);
    for (const auto& file : files)
    {
        std::ifstream ifs(file);
        std::string line;
        std::map<int, bool> already_seen_classes;
        while (std::getline(ifs, line))
        {
            groud_truth gt;
            std::istringstream iss(line);
            //class_name, left, top, right, bottom
            iss >> gt.cls;
            float left, top, right, bottom;
            iss >> left >> top >> right >> bottom;
            gt.bbox.push_back(left);
            gt.bbox.push_back(top);
            gt.bbox.push_back(right);
            gt.bbox.push_back(bottom);
            ground_truths.push_back(gt);

            if (info.gt_counter_per_class.find(gt.cls) == info.gt_counter_per_class.end())
                info.gt_counter_per_class[gt.cls] = 1;
            else
                info.gt_counter_per_class[gt.cls] += 1;

            if (already_seen_classes.find(gt.cls) == already_seen_classes.end())
            {
                if (info.counter_images_per_class.find(gt.cls) == info.counter_images_per_class.end())
                    info.counter_images_per_class[gt.cls] = 1;
                else
                    info.counter_images_per_class[gt.cls] += 1;
                already_seen_classes[gt.cls] = true;
            }
        }

    }
    return std::make_pair(ground_truths, info);
}


std::vector<std::pair<int, float>> calc_map(const std::string ground_truth_path, 
    const std::string detection_results_path)
{
    auto gt = get_ground_truth(ground_truth_path);
    auto gt_info = gt.second;
    for (const auto& e : gt_info.gt_counter_per_class)
        std::cout << e.first << " " << e.second << std::endl;
    for (const auto& e : gt_info.counter_images_per_class)
        std::cout << e.first << " " << e.second << std::endl;

    std::vector<std::pair<int, float>> mAP;
    mAP.push_back(std::make_pair(1, 0.5));
    return mAP;
}