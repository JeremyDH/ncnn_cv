//
// Created by dWX1185603 on 2023/6/6.
//
#include "mtcnn.h"
#include "iostream"

/**
 * MTCNN的整个流程
 * P网络：（1）输入任意大小的图片
 *       （2）获取图片的宽和高，如果宽和高的最小边长大于12，则使用阈值0.709对图像做图像金字塔，当缩放到12*12后传入P网络，
 *       （3）使用3*3或者5*5的卷积代替12*12的卷积在图片上进行滑动。当有人脸的地方就生成一个人脸框。
 *       （4）取置信度达标的图片的索引，留下置信度达标的生成框，使用坐标反算这些生成框在原图中的坐标值，并在原图中的画框
 *       （5）计算IOU, NMS去除重复框
 *       （6）将留下的预测框在原图上做正方形后对图像进行剪裁，缩放到24*24.输入R网络
 * R网络：（1）将置信度达标的框的坐标，反算回其在原图中的坐标值，并在原图中画框，
 *       （2）IOU,使用NMS去除重复框
 *       （3）将留下的预测框在原图上做正方形，后对图像进行裁剪，然后缩放到48*48，输入到O网络
 * O网络：
 *       （1）置信度达标的框的坐标，反算回其在原图中的坐标值，并在原图中画框，
 *       （2）通过IOU计算这些框与框之间的重合度。然后使用NMS去除重复框
 *       （3）输出最终的人脸框
 * **/


namespace mirror {
    Mtcnnface::Mtcnnface() :
            P_Net(new ncnn::Net()),
            R_Net(new ncnn::Net()),
            O_Net(new ncnn::Net()),
            pnet_size_(12),
            min_face_size(40),
            scale_factor_(0.709f),
            initialized_(false) {}

    Mtcnnface::~Mtcnnface() {
        if(P_Net){
            P_Net->clear();
        }
        if(R_Net)
        {
            R_Net ->clear();
        }
        if(O_Net)
        {
            O_Net->clear();
        }
    }
    //加载模型和权重文件

    int Mtcnnface::LoadModel(const char *root_path) {
        std::string p_param = std::string(root_path) + "/pnet.param";
        std::string p_bin = std::string(root_path) + "/pnet.bin";

        if(P_Net->load_param(p_param.c_str()) == -1 ||
           P_Net->load_model(p_bin.c_str()) == -1){
            std::cout << "Load Pnet model failed" << std::endl;

            return 10000;
        }

        std::string r_param = std::string(root_path) + "/rnet.param";
        std::string r_bin = std::string(root_path) + "/rnet.bin";

        if(R_Net->load_param(r_param.c_str()) == -1 ||
           R_Net->load_model(r_bin.c_str()) == -1){
            std::cout << "Rnet is empty" << std::endl;

            return 10000;
        }

        std::string o_param = std::string(root_path) + "/onet.param";
        std::string o_bin = std::string(root_path) + "/onet.bin";

        if(O_Net->load_param(o_param.c_str()) == -1 ||
           O_Net->load_model(o_bin.c_str()) == -1){
            std::cout << "Onet is empty" << std::endl;
        }
        initialized_ = true;
        return 0;
    }

    int Mtcnnface::DetectFace(const cv::Mat &img_src, std::vector<FaceInfo> *faces) {
        //start detect
        if(img_src.empty()){
            std::cout << "input image is empty" << std::endl;
            return 10001;
        }
        if(!initialized_){
            std::cout << "model is empty" << std::endl;
            return 10000;
        }
        //获取输入特征的信息
        cv::Size max_size = cv::Size(img_src.cols, img_src.rows);
        cv::Mat img_cpy = img_src.clone();
        //将cv::mat 转化为ncnn::mat
        ncnn::Mat img_in = ncnn::Mat::from_pixels(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                          img_src.cols, img_src.rows);
        img_in.substract_mean_normalize(meanVals, normVals);

        //定义三个阶段输出的bbox，并对其进行存储
        std::vector<FaceInfo> first_bbox, second_bbox;
        std::vector<FaceInfo> first_bbox_result;

        std::cout << "PDetect is start" << std::endl;
        PDetect(img_in, &first_bbox);
        NMS(first_bbox, &first_bbox_result, nms_threshold_[0]);
        Refine(&first_bbox_result, max_size);

        std::cout << "RDetect is start" << std::endl;
        RDetect(img_in, first_bbox_result, &second_bbox);
        std::vector<FaceInfo> second_bbox_result;
        NMS(second_bbox, &second_bbox_result, nms_threshold_[1]);
        Refine(&second_bbox_result, max_size);

        std::cout << "ODetect is start" << std::endl;
        std::vector<FaceInfo> third_bbox_result;
        ODetect(img_in, second_bbox_result, &third_bbox_result);
        NMS(third_bbox_result, faces, nms_threshold_[2]);
        Refine(faces, max_size);

        return 0;
    }
    int Mtcnnface::PDetect(const ncnn::Mat &img_in, std::vector<FaceInfo> *first_bboxes) {

        first_bboxes->clear();
        int width = img_in.w;
        int height = img_in.h;
        float min_side = MIN(width, height);
        float curr_scale = float(pnet_size_) / min_face_size;
        min_side *= curr_scale;
        std::vector<float> scales;
        while(min_side > pnet_size_)
        {
            scales.push_back(curr_scale);
            min_side *= scale_factor_;
            curr_scale *= scale_factor_;
        }
        //获取图片金字塔,按照不同缩放系数改变图片大小
        for(int i = 0; i < static_cast<size_t>(scales.size()); ++i)
        {
            int w = static_cast<int>(width * scales[i]);
            int h = static_cast<int>(height * scales[i]);
            ncnn::Mat img_resized;
            ncnn::resize_bilinear(img_in, img_resized, w, h);
            ncnn::Extractor ex = P_Net->create_extractor();

            ex.set_light_mode(true);
            ex.input("data", img_resized);
            ncnn::Mat score_mat, location_mat;
            ex.extract("prob1", score_mat);
            ex.extract("conv4-2", location_mat);
            const int stride = 2;
            const int cell_size = 12;
            for(int h = 0; h < score_mat.h; ++h){
                for(int w = 0; w < score_mat.w; ++w){
                    int index = h * score_mat.w + w;
                    float score = score_mat.channel(1)[index];
                    if(score < threshold_[0]){
                        continue;
                    }

                    int x1 = round((stride * w + 1) / scales[i]);
                    int y1 = round((stride * h + 1) / scales[i]);
                    int x2 = round((stride * w + 1 + cell_size) / scales[i]);
                    int y2 = round((stride * h + 1 + cell_size) / scales[i]);

                    //regression bounding box
                    float x1_reg = location_mat.channel(0)[index];
                    float y1_reg = location_mat.channel(1)[index];
                    float x2_reg = location_mat.channel(2)[index];
                    float y2_reg = location_mat.channel(3)[index];

                    int bbox_width = x2 - x1 + 1;
                    int bbox_height = y2 - y1 + 1;

                    FaceInfo face_info;
                    face_info.score_ = score;
                    face_info.location_.x = x1 + x1_reg * bbox_width;
                    face_info.location_.y = y1 + y1_reg * bbox_height;
                    face_info.location_.width = x2 + x2_reg * bbox_width -  face_info.location_.x;
                    face_info.location_.height = y2 + y2_reg * bbox_height - face_info.location_.y;
                    face_info.location_ = face_info.location_ & cv::Rect(0,0, width, height);
                    first_bboxes->push_back(face_info);
                }
            }

        }
        return 0;
    }


    int Mtcnnface::RDetect(const ncnn::Mat &img_in, const std::vector<FaceInfo> &first_bboxes,
                           std::vector<FaceInfo> *second_bboxes) {

        second_bboxes->clear();
        for(int i = 0; i < static_cast<int>(first_bboxes.size()); ++i)
        {
            //rect = rect1 & rect2 计算交集   rect1 | rect2 计算并集
            cv::Rect face = first_bboxes.at(i).location_ & cv::Rect(0, 0, img_in.w, img_in.h);
            ncnn::Mat img_resize, img_face;
            //rect.br()返回右下角的坐标，rect.tl()返回左上角坐标
            ncnn::copy_cut_border(img_in, img_face, face.y, img_in.h - face.br().y, face.x, img_in.w - face.br().x);
            ncnn::resize_bilinear(img_face, img_resize, 24, 24);

            ncnn::Extractor ex = R_Net->create_extractor();

            ex.set_light_mode(true);
            ex.set_num_threads(2);
            ex.input("data", img_resize);
            ncnn::Mat score_mat, location_mat;
            ex.extract("prob1", score_mat);
            ex.extract("conv5-2", location_mat);
            float score = score_mat[1];
            if (score < threshold_[1]) continue;
            float x_reg = location_mat[0];
            float y_reg = location_mat[1];
            float w_reg = location_mat[2];
            float h_reg = location_mat[3];

            FaceInfo face_info;
            face_info.score_ = score;
            face_info.location_.x = face.x + x_reg * face.width;
            face_info.location_.y = face.y + y_reg * face.height;
            face_info.location_.width = face.x + face.width +
                    w_reg * face.width -  face_info.location_.x;
            face_info.location_.height = face.y + face.height +
                    h_reg * face.height - face_info.location_.y;
            second_bboxes->push_back(face_info);

        }
        return 0;
    }

    int Mtcnnface::ODetect(const ncnn::Mat &img_in, const std::vector<FaceInfo> &second_bboxes,
                           std::vector<FaceInfo> *third_bboxes) {
        third_bboxes->clear();
        for(int i = 0; i < static_cast<int>(second_bboxes.size()); ++i)
        {
            cv::Rect face = second_bboxes.at(i).location_ & cv::Rect(0, 0, img_in.w, img_in.h);
            ncnn::Mat img_face, img_resize;
            ncnn::copy_cut_border(img_in, img_face, face.y, img_in.h - face.br().y, face.x, img_in.w - face.br().x);
            ncnn::resize_bilinear(img_face, img_resize, 48, 48);

            ncnn::Extractor ex = O_Net->create_extractor();
            ex.set_light_mode(true);
            ex.set_num_threads(2);
            ex.input("data", img_resize);
            ncnn::Mat score_mat, location_mat, keypoints_mat;
            ex.extract("prob1", score_mat);
            ex.extract("conv6-2", location_mat);
            ex.extract("conv6-3", keypoints_mat);

            float score = score_mat[1];
            if ( score < threshold_[1]) continue;
            float x_reg = location_mat[0];
            float y_reg = location_mat[1];
            float w_reg = location_mat[2];
            float h_reg = location_mat[3];

            FaceInfo face_info;
            face_info.score_ = score;
            face_info.location_.x = face.x + x_reg * face.width;
            face_info.location_.y = face.y + y_reg * face.height;
            face_info.location_.width = face.x + face.width +
                    w_reg * face.width - face_info.location_.x;
            face_info.location_.height = face.y + face.height +
                    h_reg * face.height -  face_info.location_.y;

            for(int num=0; num<5; num++)
            {
                face_info.keypoints_[num] = face.x + face.width * keypoints_mat[num];
                face_info.keypoints_[num] = face.y + face.height * keypoints_mat[num+5];
            }

            third_bboxes->push_back(face_info);

        }
        return 0;
    }


    int Mtcnnface::Refine(std::vector<FaceInfo>* bboxes, const cv::Size max_size) {
        int num_boxes = static_cast<int>(bboxes->size());
        for(int i=0; i < num_boxes; ++i)
        {
           FaceInfo face_info = bboxes->at(i);
           int width = face_info.location_.width;
           int height = face_info.location_.height;
           float max_side = MAX(width, height);

           face_info.location_.x = face_info.location_.x + 0.5 * width - 0.5 * max_side;
           face_info.location_.y = face_info.location_.y + 0.5 * height - 0.5 * max_side;
           face_info.location_.width = max_side;
           face_info.location_.height = max_side;
           face_info.location_ = face_info.location_ & cv::Rect(0, 0, max_size.width, max_size.height);

           bboxes->at(i) = face_info;
        }
        return 0;
    }
}