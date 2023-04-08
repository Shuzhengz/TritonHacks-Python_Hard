depth.convertTo(depth, CV_64FC1);

Mat nor(depth.size(), CV_64FC3);

for(int x = 1; x < depth.cols - 1; ++x)
{
   for(int y = 1; y < depth.rows - 1; ++y)
   {
      Vec3d t(x,y-1,depth.at<double>(y-1, x)/*depth(y-1,x)*/);
      Vec3d l(x-1,y,depth.at<double>(y, x-1)/*depth(y,x-1)*/);
      Vec3d c(x,y,depth.at<double>(y, x)/*depth(y,x)*/);
      Vec3d d = (l-c).cross(t-c);
      Vec3d n = normalize(d);
      nor.at<Vec3d>(y,x) = n;
   }
}

imshow("normals", nor);
