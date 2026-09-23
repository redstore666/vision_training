#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main(){
    //[0]1.1 读取与显示
    Mat img =imread("../resources/test_image.png");
    if(img.empty()){
        std::cerr<<"cannot read image\n";
        return 1;
    }

    //imshow("Original",img);

    //[1]1.2灰度
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);//输入 输出 转换代码
    //imshow("Gray",gray);
    imwrite("../result/task1_images/gray.png", gray);

    //[2,4]1.6HSV颜色空间处理
    //H色相 S饱和度 V明度
    //OpenCV 为了用 8 位存储，把 H 压缩到了 0-179，S 和 V 依然是 0-255。而在标准数学里 H 是 0-360 度
    
    //转换到HSV空间-"扣除主体"
    Mat hsv;
    cvtColor(img,hsv,COLOR_BGR2HSV);
    //提取红色（两段区间)
    Mat maskLow,maskHigh,redMask;
    //low:0-10
    inRange(hsv,Scalar(0,100,100),Scalar(10,255,255),maskLow);
    //high:170-179
    inRange(hsv,Scalar(170,100,100),Scalar(179,255,255),maskHigh);
    //合并两段掩膜
    bitwise_or(maskLow,maskHigh,redMask);
    /*
    使用要点与调参技巧
    inRange() 输出的是单通道二值掩膜（黑白图）。大于阈值范围的设为255，其余设为0。

    Scalar(H, S, V) 分别对应三个通道的下限和上限。

    阈值仅为示例，必须根据相机、曝光和环境调整。

    低饱和度或过曝区域的色相通常不可靠（比如白色物体，其 H 值会乱跳）。

    任务1提示：在作业中需要说明红色、黄色边缘和阴影的处理效果。红色提取容易出现边缘发白、阴影处变黑的情况，这就是后续学习形态学（闭运算）去修补漏洞的原因。
    */
    
    //imshow("Red Mask Low", maskLow);
    //imshow("Red Mask High", maskHigh);
    //imshow("Red Mask Merged", redMask);
    //[4]红色掩膜
    imwrite("../result/task1_images/red_mask.png", redMask);

    //[2]HSV单通道图
    Mat hsvChannels[3];
    split(hsv, hsvChannels); // 分离 H, S, V
    imwrite("../result/task1_images/hsv_h.png", hsvChannels[0]);
    imwrite("../result/task1_images/hsv_s.png", hsvChannels[1]);
    imwrite("../result/task1_images/hsv_v.png", hsvChannels[2]);
    
    //[3]1.3图像滤波
    //均值滤波 简单，但会让边缘变模糊
    Mat meanImg;
    blur(img,meanImg,Size(15,15));//核越大，平滑效果越强，图像越模糊
    //高斯滤波 最常用的平滑预处理，边缘保留比均值好一点。
    Mat gaussianImg;
    GaussianBlur(img,gaussianImg,Size(5,5),1.5);//1.5-权重的分散程度
    //中值滤波 非常适合去除椒盐噪声（孤立的黑白点）+保留边缘
    Mat medianImg;
    medianBlur(img,medianImg,5);//中值滤波的核尺寸。必须大于 1 的奇数（如 3, 5, 7, 9）
    //show
    //imshow("Mean",meanImg);
    //imshow("Gaussian",gaussianImg);
    //imshow("Median",medianImg);
    imwrite("../result/task1_images/mean_filter.png", meanImg);
    imwrite("../result/task1_images/gaussian_filter.png", gaussianImg);
    imwrite("../result/task1_images/median_filter.png", medianImg);
    
    //1.4图像二值化-得到目标区域（面）
    //全局阈值-better目标与背景亮度差异大的场景
    Mat binary;
    threshold(gray,binary,128,255,THRESH_BINARY);//大于阈值的像素设为 255，其余设为 0
    //自适应阈值-要求输入必须是 8位单通道（灰度图）-better光照不均匀的图像
    Mat adaptive;
    adaptiveThreshold(gray,adaptive, 255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,2);
    //👆 参数说明：最大值为255，自适应方法为均值，阈值类型为二值，窗口大小为11(必须是>1的奇数，常数C为2，两个数值根据光照调整
    //如果图片光照不均，全局阈值会把有阴影的部分全部变黑，而自适应阈值能更好地保留细节
    //imshow("Global Threshold",binary);
    //imshow("Adaptive Threshold", adaptive);

    //1.5 Canny边缘检测-得到变化边界（线）
    Mat smooth, edges;//平滑->Canny
    GaussianBlur(gray,smooth,Size(5,5),1.5);
    Canny(smooth,edges,100,200);//100是low阈值，200high
    //高阈值选出强边缘，与强边缘相连的弱边缘可被保留
    //如果边缘太多，调高阈值；如果边缘断了，调低阈值
    //imshow("Canny",edges);

    




    //1.7直方图均衡化-增强整体对比度-会把噪声也放大
    Mat equalized;
    equalizeHist(gray,equalized);//必须输入 8 位单通道灰度图
    //imshow("Equalized", equalized);

    /*扩展：彩色图像对比度增强（仅处理v通道)
    Mat hsv2,channels[3];
    cvtColor(img,hsv2,COLOR_BGR2HSV);
    split(hsv2,channels);
    equalizeHist(channels[2],channels[2]);
    merge(channels,3,hsv2);
    Mat enhancedImg;
    cvtColor(hsv2,enhancedImg,COLOR_HSV2BGR);
    //imshow("Enhanced Color", enhancedImg);
    */

    //[5]1.8形态学操作（利用1.7redMask)
    //定义结构元素（核）
    // 常用形状：MORPH_RECT(矩形), MORPH_ELLIPSE(椭圆), MORPH_CROSS(十字)
    Mat kernel=getStructuringElement(MORPH_RECT,Size(5,5));
    //基本操作
    Mat dilated,eroded;
    dilate(redMask,dilated,kernel);//膨胀 
    erode(redMask,eroded,kernel);//腐蚀
    //开闭运算-开运算（先腐后膨）-闭运算（先膨后腐）
    Mat opened,closed;
    morphologyEx(redMask,opened,MORPH_OPEN,kernel);
    morphologyEx(redMask,closed,MORPH_CLOSE,kernel);
    //imshow("Dilated", dilated);
    //imshow("Eroded", eroded);
    //imshow("Opened", opened);
    //imshow("Closed", closed);
    imwrite("../result/task1_images/erode.png", eroded);
    imwrite("../result/task1_images/dilate.png", dilated);
    imwrite("../result/task1_images/open.png", opened);
    imwrite("../result/task1_images/close.png", closed);


    //1.9 Sobel算子（计算梯度）
    Mat gradX,gradY,absX,absY,sobelCombined;
    // 求 X 方向梯度（用 CV_16S 保留负值）
    Sobel(gray,gradX,CV_16S,1,0);
    // 求 Y 方向梯度
    Sobel(gray, gradY, CV_16S, 0, 1);
    //取绝对值并转换为8位无符号显示
    convertScaleAbs(gradX,absX);
    convertScaleAbs(gradY,absY);
    //合并X和Y边缘
    addWeighted(absX,0.5,absY,0.5,0,sobelCombined);
    //imshow("Sobel X", absX);
    //imshow("Sobel Y", absY);
    //imshow("Sobel Combined", sobelCombined);

    /*//1.10颜色边缘检测-实战不推荐(此处hsvChannels在上面声明了，会重复声明)
    Mat hsvChannels[3], hueEdges;
    split(hsv, hsvChannels); // 分离 H, S, V
    Canny(hsvChannels[0], hueEdges, 100, 200); // 对 H 通道做 Canny
    //imshow("Hue edges", hueEdges);*/

    //1.11 从边缘到轮廓（重点！）
    // 先对 redMask 做闭运算，把断裂的地方缝合起来
    Mat morph;
    Mat kernel2=getStructuringElement(MORPH_RECT,Size(3,3));
    morphologyEx(redMask,morph,MORPH_CLOSE,kernel2);
    //查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // 参数说明：输入二值图，输出轮廓，层级关系，模式，近似方法
    findContours(morph,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    cout<<"Found"<<contours.size()<<"contours."<<endl;
    // 可选：在原图上画出所有轮廓看看
    //Mat contourImg= img.clone();
    //drawContours(contourImg,contours,-1,Scalar(0,255,0),2);// -1表示画所有轮廓
    //imshow("All Contours",contourImg);

    //[6]1.12 遍历，筛选，画框
    Mat result = img.clone();//在原图副本上画
    for (size_t i=0;i< contours.size();i++){
        double area=contourArea(contours[i]);
        //筛选1:过滤掉面积太小的噪点（数值可调）
        if(area<500.0) continue;

        Rect box=boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width)/box.height;
        // 筛选2：过滤掉长宽比离谱的（比如细长的噪声）
        if(ratio<0.2||ratio>5.0) continue;
        //画外接矩形（红色框）
        rectangle(result,box,Scalar(0,0,255),2);
        //画轮廓（绿色线)
        drawContours(result,contours,static_cast<int>(i),Scalar(0,255,0),2);
        // 可选：在框上标明面积
        // string text = "Area: " + to_string((int)area);
        // putText(result, text, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        
    }
    //imshow("Filtered contours", result);
    imwrite("../result/task1_images/contours_boxes.png", result);


    //1.13绘制与变换
    //[7]绘制圆、矩形和文字
    Mat drawImg = img.clone();
    circle(drawImg, Point(100, 100), 50, Scalar(255, 0, 0), 2); // 画圆，蓝色
    rectangle(drawImg, Point(200, 200), Point(400, 400), Scalar(0, 255, 0), 2); // 画矩形，绿色
    putText(drawImg, "Hello OpenCV", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2); // 写文字，红色
    imwrite("../result/task1_images/drawing.png", drawImg);

    //[8]绕图像中心旋转35度
    Mat rotated;
    Point2f center(img.cols / 2.0, img.rows / 2.0); // 图像中心
    Mat rotMat = getRotationMatrix2D(center, 35.0, 1.0); // 旋转矩阵，35度，缩放比例1.0
    warpAffine(img, rotated, rotMat, img.size()); // 仿射变换
    imwrite("../result/task1_images/rotated_35deg.png", rotated);

    //[9]单独裁剪原图左上角1/4（即原宽、原高各取一半）
    Mat cropImg = img(Rect(0, 0, img.cols / 2, img.rows / 2));
    imwrite("../result/task1_images/crop_top_left.png", cropImg);

    cout << "Task 1 images saved to result/task1_images/ successfully!" << endl;


    waitKey(0);
    return 0;
}