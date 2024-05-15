#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <dirent.h>
#include <algorithm>    
#include <numeric>
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

    std::sort(files.begin(), files.end());
    return files;
}

struct bbox
{
    int cls;
    float score;
    std::vector<float> xyxy;
    bool used;
};

struct groud_truth
{
    int file_id;
    std::vector<bbox> bboxes;
};

struct groud_truth_info
{
    std::map<int, int> gt_counter_per_class;
};


std::pair<std::vector<groud_truth>, groud_truth_info> get_ground_truths(const std::string ground_truths_path)
{
    std::vector<groud_truth> ground_truths;
    groud_truth_info info;
    std::vector<std::string> files = get_files_in_path(ground_truths_path);
    int file_id = 0;
    for (const auto& file : files)
    {
        groud_truth gt;
        gt.file_id = file_id;
        std::ifstream ifs(file);
        std::string line;
        while (std::getline(ifs, line))
        {
            std::istringstream iss(line);
            //class_name, left, top, right, bottom
            int cls;
            iss >> cls;
            float left, top, right, bottom;
            iss >> left >> top >> right >> bottom;
            bbox b = { cls, 1, {left, top, right, bottom}, false };
            gt.bboxes.push_back(b);

            if (info.gt_counter_per_class.find(cls) == info.gt_counter_per_class.end())
                info.gt_counter_per_class[cls] = 1;
            else
                info.gt_counter_per_class[cls] += 1;
        }
        ground_truths.push_back(gt);
        file_id++;
    }
    return std::make_pair(ground_truths, info);
}

struct detection_result
{
    int file_id;//idx in ground_truths, must make sure the order is the same
    std::vector<bbox> bboxes;
};

struct detection_result_info
{
    std::map<int, int> dt_counter_per_class;
};

std::pair<std::vector<detection_result>, detection_result_info> get_detection_results(const std::string detection_results_path)
{
    std::vector<detection_result> detection_results;
    detection_result_info info;
    std::vector<std::string> files = get_files_in_path(detection_results_path);
    int file_id = 0;
    for (const auto& file : files)
    {
        detection_result dt;
        dt.file_id = file_id;
        std::ifstream ifs(file);
        std::string line;
        while (std::getline(ifs, line))
        {
            std::istringstream iss(line);
            //class_name, score, left, top, right, bottom
            int cls;
            iss >> cls;
            float score;
            iss >> score;
            float left, top, right, bottom;
            iss >> left >> top >> right >> bottom;
            bbox b = { cls, score, {left, top, right, bottom} };
            dt.bboxes.push_back(b);

            if (info.dt_counter_per_class.find(cls) == info.dt_counter_per_class.end())
                info.dt_counter_per_class[cls] = 1;
            else
                info.dt_counter_per_class[cls] += 1;
        }
        detection_results.push_back(dt);
        file_id++;
    }
    return std::make_pair(detection_results, info);
}

float voc_ap(std::vector<float> rec, std::vector<float> prec)
{
    //first append sentinel values at the end
    rec.insert(rec.begin(), 0);
    rec.push_back(1);
    prec.insert(prec.begin(), 0);
    prec.push_back(0);

    //compute the precision envelope
    for (int i = prec.size() - 2; i >= 0; i--)
        prec[i] = std::max(prec[i], prec[i + 1]);

    std::vector<int> i_list;
    for (int i = 1; i < rec.size(); i++)
        if (rec[i] != rec[i - 1])
            i_list.push_back(i);

    //compute area under the curve
    float ap = 0;
    for (int i = 0; i < i_list.size(); i++) {
        auto idx = i_list[i];
        ap += (rec[idx] - rec[idx - 1]) * prec[idx];

    }
    return ap;

}

#define MINOVERLAP 0.5
float iou(bbox a, bbox b)
{
    std::vector<float> bi = {
            std::max(a.xyxy[0],b.xyxy[0]), 
            std::max(a.xyxy[1],b.xyxy[1]), 
            std::min(a.xyxy[2],b.xyxy[2]), 
            std::min(a.xyxy[3],b.xyxy[3])};
    float iw = bi[2] - bi[0] + 1;
    float ih = bi[3] - bi[1] + 1;
    if ((iw > 0) && (ih > 0)) {
        float ua = (a.xyxy[2] - a.xyxy[0] + 1) * (a.xyxy[3] - a.xyxy[1] + 1) + 
             (b.xyxy[2] - b.xyxy[0] + 1) * (b.xyxy[3] - b.xyxy[1] + 1) - iw * ih;
        float ov = iw * ih / ua;
        return ov;
    }
    return 0;
}

std::vector<std::pair<int, float>> calc_mAP_(std::vector<groud_truth>& gts, groud_truth_info gt_info,
    std::vector<detection_result>& dts, detection_result_info dt_info)
{
    std::vector<int> gt_classes;
    for (const auto& e : gt_info.gt_counter_per_class)
        gt_classes.push_back(e.first);

    std::vector<std::pair<int, float>> mAP;
    for (auto cls : gt_classes) {
        int nd = dt_info.dt_counter_per_class[cls];
        if (nd == 0) {
            mAP.push_back(std::make_pair(cls, 0));
            continue;
        }
        std::vector<float> tp(nd, 0), fp(nd, 0), score(nd, 0);

        int pos = 0;
        int fp_sum = 0;
        int tp_sum = 0;
        for (int i = 0; i < gts.size(); i++) {
            auto gt = gts[i];
            auto dt = dts[i];
            for (int j = 0; j < dt.bboxes.size(); j++) {
                auto db = dt.bboxes[j];
                if (db.cls != cls)
                    continue;
                float ovmax = -1;
                int kmax = -1;
                for (int k = 0; k < gt.bboxes.size(); k++) {
                    auto gb = gt.bboxes[k];
                    if (gb.cls != cls)
                        continue;
                    float ov = iou(db, gb);
                    if (ov > ovmax) {
                        ovmax = ov;
                        kmax = k;
                    }
                }
                if (ovmax >= MINOVERLAP) {
                    if (!gt.bboxes[kmax].used) {
                        tp[pos] = 1;
                        tp_sum += 1;
                        gt.bboxes[kmax].used = true;
                    } else {
                        fp[pos] = 1;
                        fp_sum += 1;
                    } 
                } else {
                    fp[pos] = 1;
                    fp_sum += 1;
                }
                score[pos] = db.score;
                pos += 1;
            }
        }

        //sort tp/fp by score
        std::vector<int> idx(nd);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&score](int i1, int i2) {return score[i1] > score[i2]; });
        std::vector<float> tp_sorted(nd), fp_sorted(nd);
        for (int i = 0; i < nd; i++) {
            tp_sorted[i] = tp[idx[i]];
            fp_sorted[i] = fp[idx[i]];
        }
        for (int i = 0; i < nd; i++) {
            tp[i] = tp_sorted[i];
            fp[i] = fp_sorted[i];
        }

        std::vector<float> rec(nd, 0), prec(nd, 0);

        float cumsum = 0;
        for (int i = 0; i < nd; i++) {
            auto val = tp[i];
            tp[i] += cumsum;
            cumsum += val;
        }
        cumsum = 0;
        for (int i = 0; i < nd; i++) {
            auto val = fp[i];
            fp[i] += cumsum;
            cumsum += val;
        }

        for (int i = 0; i < nd; i++) {
            if (gt_info.gt_counter_per_class[cls] > 0)
                rec[i] = tp[i] / gt_info.gt_counter_per_class[cls];
            if (tp[i] + fp[i] > 0)
                prec[i] = tp[i] / (fp[i] + tp[i]);
        }

        auto ap = voc_ap(rec, prec);
        mAP.push_back(std::make_pair(cls, ap));
    }

    return mAP;
}

std::vector<std::pair<int, float>> calc_mAP(const std::string ground_truths_path, 
    const std::string detection_results_path)
{
    auto gts = get_ground_truths(ground_truths_path);
    std::cout << "gt_info in " << gts.first.size() << " files" << std::endl;
    for (const auto& e : gts.second.gt_counter_per_class)
        std::cout << "    " << e.first << " " << e.second << std::endl;
    

    auto dts = get_detection_results(detection_results_path);
    std::cout << "dt_info in " << dts.first.size() << " files" << std::endl;
    for (const auto& e : dts.second.dt_counter_per_class)
        std::cout << "    " << e.first << " " << e.second << std::endl;

    auto mAP = calc_mAP_(gts.first, gts.second, dts.first, dts.second);
    return mAP;
}