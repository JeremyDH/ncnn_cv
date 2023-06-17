//
// Created by dWX1185603 on 2023/6/13.
//

#include "yolov5.h"

namespace mirror{

    static inline float intersection_area(const ObjectInfo& a, const ObjectInfo& b)
    {
        cv::Rect_<float> inter = a.location_ & b.location_;
        return inter.area();
    }

    static void qsort_descet_inplace(std::vector<ObjectInfo>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].score_;

        while(i<=j)
        {
            while(faceobjects[i].score_ < p)
                i++;
            while(faceobjects[j].score_ < p)
                j--;
            if(i <= j)
            {
                std::swap(faceobjects[i], faceobjects[j]);
                i++;
                j--;
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                if(left < j) mirror::qsort_descet_inplace(faceobjects, left, j);
            }
#pragma omp section
            {
                if(i < right) mirror::qsort_descet_inplace(faceobjects, i, right);
            }
        }
    }

    static void qsort_descent_inplace(std::vector<ObjectInfo>& faceobjects)
    {
        if(faceobjects.empty())
            return;

        mirror::qsort_descet_inplace(faceobjects, 0, faceobjects.size()-1);
    }


    static void nms_sorted_bboxes(const std::vector<ObjectInfo>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic=false)
    {
        picked.clear();
        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for(int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].location_.area();
        }

        for(int i = 0; i < n; i++)
        {
            const ObjectInfo& a = faceobjects[i];
            int keep = 1;
            //pick中存储的候选框，a是目标框， b是候选框
            for(int j = 0; j < (int)picked.size(); j++)
            {
                //依此选候选框
                const ObjectInfo& b = faceobjects[picked[j]];

                if(!agnostic && a.label != b.label)
                    continue;
                //计算交并比
                //计算交集
                float insect_area = intersection_area(a, b);
                //计算并集
                float union_area = areas[i] + areas[picked[j]] - insect_area;
                //计算nms
                if(insect_area / union_area > nms_threshold)
                    keep = 0;
            }
            if(keep)
                picked.push_back(i);
        }
    }

    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<ObjectInfo>& objects)
    {
        //预测的表格
        const int num_grid = feat_blob.h;
        int num_grid_x;
        int num_grid_y;
        if(in_pad.w > in_pad.h)
        {
            num_grid_x = in_pad.w / stride;
            num_grid_y = num_grid / num_grid_x;
        }
        else
        {
            num_grid_y = in_pad.h / stride;
            num_grid_x = num_grid / num_grid_y;
        }

        const int num_class = feat_blob.w - 5;
        const int num_anchors = anchors.w / 2;

        for(int q = 0; q < num_anchors; q++)
        {
            const float anchor_w = anchors[q * 2];
            const float anchor_h = anchors[q * 2 + 1];

            const ncnn::Mat feat = feat_blob.channel(q);

            for(int i = 0; i < num_grid_y; i++)
            {
                for(int j=0; j < num_grid_x; j++)
                {
                    const float* featptr = feat.row(i * num_grid_x + j);
                    float box_confidence = sigmoid(featptr[4]);
                    if(box_confidence >= prob_threshold)
                    {
                        int class_index = 0;
                        float class_score = -FLT_MAX;
                        for(int k = 0; k < num_class; k++)
                        {
                            float score = featptr[5 + k];
                            if(score > class_score)
                            {
                                class_index = k;
                                class_score = score;
                            }
                        }
                        float confidence = box_confidence * sigmoid(class_score);
                        if(confidence >= prob_threshold)
                        {
                            float dx = sigmoid(featptr[0]);
                            float dy = sigmoid(featptr[1]);
                            float dw = sigmoid(featptr[2]);
                            float dh = sigmoid(featptr[3]);

                            float pb_cx = (dx * 2.f - 0.5f) * stride;
                            float pb_cy = (dy * 2.f - 0.5f) * stride;

                            float pb_w = pow(sigmoid(dw) * 2.f, 2) * stride;
                            float pb_h = pow(sigmoid(dh) * 2.f, 2) * stride;

                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;

                            ObjectInfo obj;
                            obj.location_.x = x0;
                            obj.location_.y = y0;
                            obj.location_.width = x1 - x0;
                            obj.location_.height = y1 - y0;
                            obj.label = class_index;
                            obj.score_ = confidence;

                            objects.push_back(obj);
                        }
                    }
                }
            }
        }

    }

    Yolov5::Yolov5() {
        yolov5 = new ncnn::Net();
        intialized_ = false;
    }

    Yolov5::~Yolov5() {
        if(yolov5){
            delete yolov5;
            yolov5 = nullptr;
        }
    }

    int Yolov5::Loadmodel(const char *root_path) {

        yolov5->opt.use_vulkan_compute = true;
        std::string yolov5_param = std::string(root_path) + "/yolov5s_6.0.param";
        std::string yolov5_bin = std::string(root_path) + "/yolov5s_6.0.bin";

#if YOLOV5_V62
        if(yolov5->load_param(yolov5_param.c_str()) == -1 ||
           yolov5->load_model(yolov5_bin.c_str()) == -1)
        {
            std::cout << "load model is failed"  << std::endl;
            return 10000;
        }

//        yolov5->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
//
//        if (yolov5->load_param(yolov5_param.c_str() ) != -1 ||
//            yolov5->load_model(yolov5_bin != -1))
//        {
//            std::cout << "load model is failed"  << std::endl;
//            return 10000;
//        }


        intialized_ = true;
        return 0;


//#elif YOLOV5_V60
//        if (yolov5->load_param(yolov5_param.c_str()))
//        exit(-1);
//    if (yolov5->load_model(yolov5_bin.c_str()))
//        exit(-1);
//#else
//    yolov5->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
//
//    if (yolov5->load_param("yolov5s.param"))
//        exit(-1);
//    if (yolov5.load_model("yolov5s.bin"))
//        exit(-1);


#endif

    }

    int Yolov5::Object_d(const cv::Mat &img_src, std::vector<ObjectInfo>* objects) {

        objects->clear();

        const int target_size = 640;
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        int img_w = img_src.cols;
        int img_h = img_src.rows;

        //letterbox pad to multiple of MAX_STRIDE

        int w = img_w;
        int h = img_h;
        float scale = 1.f;
        if(w > h)
        {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float) target_size / h;
            h = target_size;
            w = w * scale;
        }
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

        int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
        int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov5->create_extractor();
        ex.input("images", in_pad);

        std::vector<ObjectInfo> proposals;

        //stride 8
        {
            ncnn::Mat out;
            ex.extract("Transpose_606", out);
            ncnn::Mat anchors(6);
            anchors[0] = 10.f;
            anchors[1] = 13.f;
            anchors[2] = 16.f;
            anchors[3] = 30.f;
            anchors[4] = 33.f;
            anchors[5] = 23.f;

            std::vector<ObjectInfo> objects8;
            mirror::generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        //stride 16
        {
            ncnn::Mat out;

#if YOLOV5_V62
            ex.extract("353", out);
#elif YOLOV5_V60
            ex.extract("376", out);
#else
        ex.extract("781", out);
#endif

            ex.extract("output", out);
            ncnn::Mat anchors(6);
            anchors[0] = 30.f;
            anchors[1] = 61.f;
            anchors[2] = 62.f;
            anchors[3] = 45.f;
            anchors[4] = 59.f;
            anchors[5] = 119.f;

            std::vector<ObjectInfo> objects16;
            mirror::generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);
            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        //对预测框进行排序
        mirror::qsort_descent_inplace(proposals);
        std::vector<int> picked;
        // 使用极大值抑制
        mirror::nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();
        objects->resize(count);

        std::vector<ObjectInfo> object_temp;
        object_temp.resize(count);

        for(int i = 0; i < count; i++)
        {
            object_temp[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (object_temp[i].location_.x - (wpad / 2)) / scale;
            float y0 = (object_temp[i].location_.y - (hpad / 2)) / scale;
            float x1 = (object_temp[i].location_.x + object_temp[i].location_.width - (wpad / 2)) / scale;
            float y1 = (object_temp[i].location_.y + object_temp[i].location_.height - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            object_temp[i].location_.x = x0;
            object_temp[i].location_.y = y0;
            object_temp[i].location_.width = x1 - x0;
            object_temp[i].location_.height = y1 - y0;

        }
        objects = &object_temp;

        return 0;
    }


}
