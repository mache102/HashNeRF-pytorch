#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <chrono>

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void eul2rot(Mat& R, double roll, double pitch, double yaw) {
    Mat R_x = (Mat_<double>(3, 3) << 
               1, 0,         0, 
               0, cos(roll), -sin(roll), 
               0, sin(roll), cos(roll));

    Mat R_y = (Mat_<double>(3, 3) << 
               cos(pitch),  0, sin(pitch), 
               0,           1, 0, 
               -sin(pitch), 0, cos(pitch));

    Mat R_z = (Mat_<double>(3, 3) << 
               cos(yaw), -sin(yaw), 0, 
               sin(yaw), cos(yaw),  0, 
               0,        0,         1);

    R = R_x * R_y * R_z;
}

// rotate pixel, in_vec as input(row, col)
Vec2i rotate_pixel(const Vec2i& in_vec, Mat& rot_mat, int width, int height)
{
    Vec2d vec_rad = Vec2d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width);

    Vec3d vec_cartesian;
    vec_cartesian[0] = -sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = cos(vec_rad[0]);

    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian[0] + rot_mat_data[1]*vec_cartesian[1] + rot_mat_data[2]*vec_cartesian[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian[0] + rot_mat_data[4]*vec_cartesian[1] + rot_mat_data[5]*vec_cartesian[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian[0] + rot_mat_data[7]*vec_cartesian[1] + rot_mat_data[8]*vec_cartesian[2];

    Vec2d vec_rot;
    vec_rot[0] = acos(vec_cartesian_rot[2]);
    vec_rot[1] = atan2(vec_cartesian_rot[1], -vec_cartesian_rot[0]);
    if(vec_rot[1] < 0)
        vec_rot[1] += M_PI*2;

    Vec2i vec_pixel;
    vec_pixel[0] = height*vec_rot[0]/M_PI;
    vec_pixel[1] = width*vec_rot[1]/(2*M_PI);

    return vec_pixel;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        cerr << "Usage: calibrate_one_image <file_path> <roll> <pitch> <yaw>" << endl;
        cerr << "<roll>, <pitch>, <yaw> are rotation angles in degrees, 0~360" << endl;
        return 1;
    }

    string filePath = argv[1];
    double roll = RAD(atof(argv[2]));
    double pitch = -RAD(atof(argv[3]));
    double yaw = -RAD(atof(argv[4]));

    Mat im = cv::imread(filePath);
    if (im.data == NULL) {
        cerr << "Unable to open image" << endl;
        return 1;
    }
    int h = im.rows;
    int w = im.cols;
    int size = h*w;

    Mat R;
    eul2rot(R, roll, pitch, yaw);

    Size im_shape(h, w);

    Mat2i im_pixel_rotate(h, w);
    Mat im_out(im.rows, im.cols, im.type());
    Vec3b* im_data = (Vec3b*)im.data;
    Vec3b* im_out_data = (Vec3b*)im_out.data;

    auto start = chrono::high_resolution_clock::now();
    
    int n = 3;
    for (int i = 0; i < static_cast<int>(h); i++) {
        for (int j = 0; j < static_cast<int>(w); j++) {
            // inverse warping
            Vec2i vec_pixel = rotate_pixel(Vec2i(i, j), R, w, h);
            int o_i = vec_pixel[0];
            int o_j = vec_pixel[1];
            // if o_i and o_j are both out of bounds (but within n pixels), use nearest neighbor
            // otherwise, only then ignore

            if ((o_i > -n) && (o_j > -n) && (o_i < h+n) && (o_j < w+n)) {
                // clamp to image size
                o_i = max(0, min(h-1, o_i));
                o_j = max(0, min(w-1, o_j));

                // if ((o_i >= 0) && (o_j >= 0) && (o_i < h) && (o_j < w))
                im_out_data[i * w + j] = im_data[o_i * w + o_j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time elapsed: " << duration.count() << " s" << endl;

    String savename = "warped_image_cpp.png";
    cout << "Save to " << savename << endl;
    imwrite(savename, im_out);

    return 0;
}