// Traffic_Sign_Detection.cpp : Defines the entry point for the console application.

#include "Traffic_Sign_Detection.h"

int main(int argc, char** argv)
{
	// loop until chose "q" on menu or result image
	while(true)
	{
		string image = "";
		Mat img_original, img_equalized,
			img_mask, img_equalized_mask,
			img_bw, img_eq_bw,
			proc_image, proc_image_eq,
			img_hough, img_hough_eq;

		/// select type of input and receve the file name
		if (menu(image) == false)
			break;
		cout << endl << "Loading input image: " << image << endl;

		// start processing the image
		img_original = imread(image, CV_LOAD_IMAGE_COLOR);
		double maxsize = max(img_original.size().width, img_original.size().height);
		double resize_width = 700*(img_original.size().width/maxsize); //width(largura)
		double resize_height = 700*(img_original.size().height/maxsize); //height(altura)
		if(DEBUG_TEX) cout << "\nResize Image to: " << resize_width << " x " << resize_height << " (width x height)" << endl;
		namedWindow( "Original", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
		resizeWindow("Original", (int) resize_width, (int) resize_height); 
		imshow("Original", img_original);

		img_equalized = image_hist_equalizer(img_original);
		proc_image = img_original.clone();
		proc_image_eq = img_equalized.clone();

		/// segmentation of image by color
		detect_color(proc_image, proc_image_eq, img_bw, img_eq_bw);

		/// detecting circles shapes
		vector<Vec3f> circles;
		hough_detection(proc_image, circles, img_hough);
		if(DEBUG_TEX) cout << "Detected " << circles.size() << " circles in original image." << endl;
		vector<Vec3f> circles_eq;
		hough_detection(proc_image_eq, circles_eq, img_hough_eq);
		if(DEBUG_TEX) cout << "Detected " << circles_eq.size() << " circles in equalizaded image." << endl;

		if(DEBUG_IMG)
		{
			namedWindow("calcHist Original", CV_WINDOW_AUTOSIZE );
			imshow("calcHist Original", histogramImage(img_original) );

			namedWindow( "Equalized", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
			resizeWindow("Equalized", (int) resize_width, (int) resize_height); 
			imshow("Equalized", img_equalized);
			
			namedWindow("calcHist Equalized", CV_WINDOW_AUTOSIZE );
			imshow("calcHist Equalized", histogramImage(img_equalized) );

			namedWindow( "Black white", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
			resizeWindow("Black white", (int) resize_width, (int) resize_height); 
			imshow("Black white", img_bw);

			namedWindow( "Black white eq", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
			resizeWindow("Black white eq", (int) resize_width, (int) resize_height); 
			imshow("Black white eq", img_eq_bw);

			namedWindow("Segmented by color", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO);
			resizeWindow("Segmented by color", (int) resize_width, (int) resize_height);
			imshow("Segmented by color", proc_image);

			namedWindow("Segmented by color equalizaded", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO);
			resizeWindow("Segmented by color equalizaded", (int) resize_width, (int) resize_height);
			imshow("Segmented by color equalizaded", proc_image_eq);

			namedWindow("Hough Circle Transform original", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
			resizeWindow("Hough Circle Transform original", (int) resize_width, (int) resize_height);
			imshow("Hough Circle Transform original", img_hough );

			namedWindow("Hough Circle Transform equalizaded", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
			resizeWindow("Hough Circle Transform equalizaded", (int) resize_width, (int) resize_height);
			imshow("Hough Circle Transform equalizaded", img_hough_eq );
		}

		circles.insert(circles.end(), circles_eq.begin(), circles_eq.end() );

		/// cropping the images within the circles
		vector<Mat> crop_signs = crop_sign(img_original, img_bw, circles);
		vector<Mat> templates;

		/// loading the templates to a vector
		for(int i = 0; i < 12; i++)
		{
			ostringstream s;
			s << "templates/temp_" << (i + 1) * 10 << ".jpg";
			templates.push_back(imread(s.str(), 1));
		}

		Mat templ, img_mask_thres, img_mask_thres_max;	
		double max_corr = 0.0;
		int signDetected = -1, max_crop_sign = -1;
		Size_number size_max_nr;
		/// For all cropped images with the right size
		for (int i = 0; i < (int) crop_signs.size(); i++){
			Mat img = crop_signs.at(i);
			Size_number size_nr = get_size_numbers(img, img_mask_thres);
			if(size_nr.height !=-1 && size_nr.width !=-1 && size_nr.pointX !=-1 && size_nr.pointY !=1){
				for (int j = 0; j < (int) templates.size(); j++){
					templ = templates.at(j);
					/// Resize the template to the right proportions
					Size s(size_nr.width, size_nr.height);
					resize(templ, templ, s, 0, 0, CV_INTER_AREA );
					double corr = matchingMethod(img, templ, j);
					if(DEBUG_TEX) cout << "detetou o " << (j + 1) * 10 << "km/h com uma precisao de " << corr << "%" << endl;
					if(corr >= max_corr)
					{
						signDetected = j;
						max_corr = corr;
						size_max_nr = size_nr;
						max_crop_sign = i;
						img_mask_thres_max = img_mask_thres;
					}
				}
				if (max_corr > 0.90)
					break;

			}
		}

		if(signDetected != -1 && max_corr >= MIN_CORRELATION_ACCEPTED)
		{
			if(DEBUG_IMG || DEBUG_IMG_CROP)
			{
				Mat img_crop = crop_signs.at(max_crop_sign);
				imshow( "crop sign", img_crop );
				imshow("crop sign bw", img_mask_thres_max);

				//Cropping the interior of the detected sign
				if(size_max_nr.pointX != -1 && size_max_nr.pointY != -1 && size_max_nr.height != -1 && size_max_nr.width != -1)
				{
					Mat imCrop = img_crop(Rect(size_max_nr.pointX, size_max_nr.pointY, size_max_nr.width, size_max_nr.height));
					imshow("croped letters only", imCrop);
				}
			} 

			//cout << "Place on the crop vector: " << max_crop_sign << endl;
			ostringstream speed, speed_file;
			speed << (signDetected + 1) * 10;
			speed_file << "signs/limite_" << speed.str() << ".png";
			cout << "Maximum speed detected: " << speed.str() << " km/h";
			namedWindow("Speed Sign Detected", CV_WINDOW_NORMAL | CV_WINDOW_FREERATIO );
			resizeWindow("Speed Sign Detected", 400, 410);
			imshow("Speed Sign Detected", imread(speed_file.str()));
		} 
		else
		{
			cout << "Unable to detect maximum speed" << endl;
		}
	
		/// handler key event on the result image
		int c = waitKey(0);
		destroyAllWindows();
		if ((char)c == 'q')
			break;
	}
	return 0;
}

Size_number get_size_numbers(Mat &img, Mat &img_mask_thres)
{
	Mat imgGrayScale, img_aux, img_bw/*, img_mask_color*/ ;
	//detect_mask(img, img_mask_color, BLACK);
	//if(DEBUG_IMG) imshow("segment bw", img_mask_color);

	cvtColor(img,imgGrayScale,CV_RGB2GRAY);
	threshold(imgGrayScale, img_mask_thres, BW_THRESHOLD, 255, CV_THRESH_BINARY_INV);
	detect_mask(img, img_aux, RED);
	bitwise_not(img_aux, img_bw);
	//imgBlackWhite &= imgBlackWhite1;
	img_mask_thres &= img_bw;
	//if(DEBUG_IMG) imshow("Grayscale threshold bw", img_mask_thres);

	int width = img_mask_thres.size().width;
	int height = img_mask_thres.size().height;

	unsigned char *input = (unsigned char*)(img_mask_thres.data);
	
	vector<double> lines, columns;
	lines.assign(height, 0);
	columns.assign(width, 0);
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			int bwColor = input[width *  j + i];
			if(bwColor == 255) {
				///getting the positions with black color
				lines.at(j) += 1;
				columns.at(i) +=1;
			}
		}
	}
	int min_letter_height=-1, max_letter_height=-1, min_letter_width=-1, max_letter_width=-1;
	bool min_height_found = false, max_height_found = false, min_width_found = false, max_width_found = false;
	int size = lines.size();
	for(int i = 0; i < size; i++) 
	{
		///Determination of the minimum/maximum position in the y axe that as letters
		if(lines.at(i) > MAX_NOISE_ACCEPTED * height && !min_height_found)
		{
			min_letter_height = i;
			min_height_found = true;
		}
		if(lines.at(size-i-1) > MAX_NOISE_ACCEPTED * height && !max_height_found)
		{
			max_letter_height = size - i - 1;
			max_height_found = true;
		}
	}
	size = columns.size();
	for(int i = 0; i < size; i++) 
	{
		///Determination of the minimum/maximum position in the x axe that as letters
		if(columns.at(i) > MAX_NOISE_ACCEPTED * width && !min_width_found)
		{
			min_letter_width = i;
			min_width_found = true;
		}
		if(columns.at(size-i-1) > MAX_NOISE_ACCEPTED * width && !max_width_found)
		{
			max_letter_width = size - i - 1;
			max_width_found = true;
		}
	}
	Size_number nr_size;
	nr_size.pointX = -1; nr_size.pointY =-1; nr_size.height =-1; nr_size.width = -1;
	if(min_letter_width < max_letter_width && min_letter_height < max_letter_height)
	{
		///Creating the Siz_number object
		nr_size.pointX = min_letter_width;
		nr_size.pointY = min_letter_height;
		nr_size.width = abs(max_letter_width - min_letter_width);
		nr_size.height = abs(max_letter_height - min_letter_height);
		if (DEBUG_TEX) cout << "width: " << nr_size.width << "height: " << nr_size.height << endl;
		//Mat imCrop;
		//Cropping the interior of the detected sign
		//imCrop = img(Rect(nr_size.pointX, nr_size.pointY, nr_size.width, nr_size.height));
		//if(DEBUG_IMG) imshow("croped letters only", imCrop);
	}
	return nr_size;
}

double matchingMethod(Mat &img, Mat &templ, int signalToDetect)
{
	/// Source image to display
	Mat img_display;
	img.copyTo( img_display );

	/// Create the result matrix
	int result_cols =  img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	Mat result;
	result.create(result_cols, result_rows, CV_32FC1 );

	/// Do the Matching and Normalize
	matchTemplate(img, templ, result, CV_TM_CCORR_NORMED);
	
	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	matchLoc = maxLoc;

	//if(DEBUG_IMG) imshow( "image_window", img_display );
	///returning the maximum correlation value found
	return maxVal;
}

vector<Mat> crop_sign(Mat &src, Mat &img_bw, vector<Vec3f> &circles)
{
	vector<Mat> croped_images_list;
	for(size_t i = 0; i < circles.size(); i++)
	{
		///Retrieving the coordinates of the center of the circle
		int x = cvRound(circles[i][0]);
		int y = cvRound(circles[i][1]);
		///Retrieving the radious of the circle
		int raio = cvRound(circles[i][2]);
		///Determinating the width and the height os the new image
		int width = (int)(sqrt(3) * raio);
		int height = raio;
		Mat imCrop;
		///Cropping the interior of the detected sign
		imCrop = src(Rect((int)(x - sqrt(3) * raio / 2), (int)(y - raio / 2), width, height));
		//if(DEBUG_IMG) imshow("Croped image " + i, imCrop);
		if(height >= MIN_SIGN_HEIGHT) 
			croped_images_list.push_back(imCrop); ///get only de fragments that don't have less height that of our templates
	}
	return croped_images_list;
}

bool hough_detection(Mat &image, vector<Vec3f> &circles, Mat &img_hough)
{
	Mat src_gray;

	/// Convert it to gray
	cvtColor( image, src_gray, CV_BGR2GRAY );

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

	/// Apply the Hough Transform to find the circles
	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, max(image.cols/10, image.rows/10), 140, 60, 0, 0 );
	//HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, max(image.cols/20, image.rows/20), 100, 50, 0, 0 );

	img_hough = image.clone();
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		/// circle center
		circle( img_hough, center, 3, Scalar(0,255,0), -1, 8, 0 );
		/// circle outline
		circle( img_hough, center, radius, Scalar(20,255,0), 4, 8, 0 );
	}
	return true;
}

bool detect_color(Mat &srcImageNameNotEqualized, Mat &srcImageNameEqualized, Mat &output_img_bw, Mat &output_img_eq_bw)
{
	Mat bw_aux, bw_aux_eq;

	detect_mask(srcImageNameNotEqualized, bw_aux, RED);
	detect_mask(srcImageNameEqualized, bw_aux_eq, RED);

	bitwise_not(bw_aux, output_img_bw);
	bitwise_not(bw_aux_eq, output_img_eq_bw);

	/// To eliminate the holes, we find the contours and fill the interiors.
    vector<vector<Point> > contours, contours_eq;
    findContours(bw_aux.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    findContours(bw_aux_eq.clone(), contours_eq, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    Mat dst1 = Mat::zeros(srcImageNameNotEqualized.size(), srcImageNameNotEqualized.type());
    Mat dst2 = Mat::zeros(srcImageNameEqualized.size(), srcImageNameEqualized.type());
    drawContours(dst1, contours, -1, Scalar::all(255), CV_FILLED);
    drawContours(dst2, contours_eq, -1, Scalar::all(255), CV_FILLED);

	/// Given the mask from the previous operation, we can extract the object using logical operation AND
	srcImageNameNotEqualized &= dst1;
	srcImageNameEqualized &= dst2;
	return true;
}

bool detect_mask(Mat &imageinput, Mat &imagemask, int color)
{
	Mat hsv;
	/// convert imagem from RGB to HSV fromat
	cvtColor(imageinput, hsv, CV_BGR2HSV);
    Mat bw_aux1, bw_aux2;

	if(color==RED)
	{
		/// thresholded image to keep red color and create a mask
		//inRange(hsv, Scalar(0, 70, 45), Scalar(8, 255, 255), bw_aux1);
		inRange(hsv, Scalar(0, 70, 45), Scalar(11, 255, 255), bw_aux1);
		inRange(hsv, Scalar(170, 70, 45), Scalar(180, 255, 255), bw_aux2);
		bitwise_or(bw_aux1, bw_aux2, imagemask); // add the both sides of color scale
		return true;
	}
	else if (color==BLACK)
	{
		/// thresholded image to keep black color and create a mask
		inRange(hsv, Scalar(0, 0, 0), Scalar(180, 255, 45), imagemask);
		return true;
	}
	return false;
}

Mat image_hist_equalizer(Mat inputimage)
{
	vector<Mat> channels; 
	Mat outputimage;
	cvtColor(inputimage, outputimage, CV_BGR2YCrCb); ///change the color image from BGR to YCrCb format
	split(outputimage,channels); ///split the image into channels
	equalizeHist(channels[0], channels[0]); ///equalize histogram on the 1st channel (Y)
	merge(channels,outputimage); ///merge 3 channels including the modified 1st channel into one image
	cvtColor(outputimage, outputimage, CV_YCrCb2BGR); ///change the color image from YCrCb to BGR format 

	return outputimage;
}

bool menu(string &image)
{
	int choice = -1;
	string video = "";

	do
	{
		system("cls");
		cout << "Menu " << endl << endl;
		cout << "1 - Image file " << endl;
		cout << "2 - Video file" << endl;
		cout << "3 - Camera Stream" << endl << endl;
		cout << "0 - Sair da app" << endl << endl;
		cout << "Selection: " ;
		cin >> choice;

		switch(choice) {
			case 1:
				system("cls");
				cout << "Select the image " << endl << endl;
				if(detect_files(IMG,image)==false)
				{
					choice = -1;
					break;
				}
				image.insert(0,"images/");
				break;

			case 2: 
				system("cls");
				cout << "Select the video " << endl << endl;
				if(detect_files(VDO,video)==false)
				{
					choice = -1;
					break;
				}
				video.insert(0,"videos/");
				image="tmp_image.png";
				cout << endl << "Press 'c' to capture a frame or 'ESC' to exit video!!! " << endl;
				if(crop_video(video,image)==false)
					choice = -1;
				break;

			case 3: 
				system("cls");
				cout << "Crop a frame " << endl << endl;
				image="tmp_image.png";
				cout << endl << "Press 'c' to capture a frame or 'ESC' to exit camera!!! " << endl;
				if(crop_stream(image)==false)
					choice = -1;
				break;

			case 0:
				return false;
				break;

			default:
				return false;
				break;
		}
	} while(choice !=0 && choice !=1 && choice !=2 && choice !=3);

	return true;
}

bool crop_video(string video, string image)
{
	Mat videoFrame;
	VideoCapture capture(video);

	///check if video file has been initialised
	if (!capture.isOpened())
	{ 
		cout << "cannot open camera";
	}

	///loop until finish video
	while (capture.read(videoFrame)) {
		try
		{
			imshow("video", videoFrame);
		}
		catch (Exception& e)
		{
			const char* err_msg = e.what();
			cout << endl << "exception caught: imshow:\n" << err_msg << endl << endl;
			destroyWindow("video");
			break;
		}

		int c = waitKey(30);
		if ( (c & 255) == 27) ///exit with "ESC"
		{
			break;
		}
		else if ((char)c == 'c') ///crop frame with "c"
		{
			destroyWindow("video");
			return imwrite(image, videoFrame);
			break;
		}
	}
	destroyWindow("video");
	return false;
}

bool crop_stream(string image)
{
	Mat cameraFrame;
	VideoCapture streamCam(0);   ///0 is the id of video device.0 if you have only one camera.

	///check if video device has been initialised
	if (!streamCam.isOpened())
	{ 
		cout << "cannot open camera";
	}

	streamCam.read(cameraFrame); /// erro da camera do carlos :p
	/// keep the loop until turn off camera
	while (streamCam.read(cameraFrame))
	{
		try
		{
			if(CARLOS_COMPUTER) rotateImage(cameraFrame); ///camera invertida do Carlos :p
			imshow("camera", cameraFrame);
		}
		catch (Exception& e)
		{
			const char* err_msg = e.what();
			cout << "exception caught: imshow:\n" << err_msg << std::endl;
			destroyWindow("camera");
			return false;
			break;
		}

		int c = waitKey(30);
		if ( (c & 255) == 27)
		{
			break;
		}
		else if ((char)c == 'c')
		{
			destroyWindow("camera");
			return imwrite(image, cameraFrame);
			break;
		}
	}
	destroyWindow("camera");
	return false;
}

bool detect_files(int type_file, string &filename)
{
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError=0;
	vector<string> file_list;
	int choice = -1, i = 1;

	/// Find the first file in the directory.
	if (type_file==IMG)
		hFind = FindFirstFile(TEXT("images\\*.*g"), &ffd);
	else if (type_file==VDO)
		hFind = FindFirstFile(TEXT("videos\\*.*"), &ffd);

	if (INVALID_HANDLE_VALUE == hFind) 
	{
		cout << "Unable to detect files.!!!" << endl;
		return false;
	}
	else
	{
		/// List all the files in the directory with some info about them.
		do
		{
			if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)){}
			else if (~(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				///convert from wide char to narrow char array
				char ch[260];
				char DefChar = ' ';
				WideCharToMultiByte(CP_ACP,0,ffd.cFileName,-1, ch,260,&DefChar, NULL);
    
				///A std:string  using the char* constructor.
				string ss(ch);
				filename = ss;
				cout << i << " - " << ss << endl;
				file_list.push_back (ss);
				i++;
			}
		}
		while (FindNextFile(hFind, &ffd) != 0);

		cout << endl << "0 - Last Menu" << endl;

		dwError = GetLastError();
		if (dwError != ERROR_NO_MORE_FILES)
		{
			cout << "Error no more files!!!" << endl;
			return false;
		}
		if(((int) file_list.size())>0)
		{
			while(choice>(int) file_list.size()||choice<0)
			{
				cout << endl << "Selection: " ;
				cin >> choice;
				if(choice==0)
					return false;
				cin.clear();
				cin.ignore();
			}
			filename = file_list.at(choice-1);
		}
		else
			return false;
	}
	FindClose(hFind);
	return true;
}

void rotateImage(Mat &image)
{
	Point2d pt(image.cols/2., image.rows/2.);
    Mat r = getRotationMatrix2D(pt, 180.0, 1.0);
    warpAffine(image, image, r, cv::Size(image.cols, image.rows));
}

Mat histogramImage(Mat &image)
{
	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( image, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
						Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
						Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
						Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
						Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
						Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
						Scalar( 0, 0, 255), 2, 8, 0  );
	}

	/// Display
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	//imshow("calcHist Demo", histImage );

	return histImage;
}
