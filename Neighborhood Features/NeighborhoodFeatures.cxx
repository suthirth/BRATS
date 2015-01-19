/*
Extract and save neighborhood features for mean, standard deviation and skewness of an image.
Two sets of 3 images with radius 1 and 3. (3x3x3 region and 7x7x7 region)
Input: 3D File in ITK readable format.  
Usage: NeighborhoodFeatures <FileDirectory/> <InputFileName> <OutputDirectory/>
Output: Mean, Standard Deviation and Skewness calculated in neighborhood of radius 1 and 3. Files are saved in <OutputDirectory> with prefix.
*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef itk::Image<unsigned short, 3> ImageType;
typedef itk::Image<float, 3> FloatImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<FloatImageType> WriterType;
typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIterator;
typedef itk::ImageRegionIterator<FloatImageType> ImageIterator;

void ExtractFeatures(ImageType::Pointer data, int r, char* argv[])
{

	//Extract Neighborhood features
	NeighborhoodIterator::RadiusType radius;
	radius.Fill(r);
	NeighborhoodIterator it(radius, data, data->GetRequestedRegion());
	
	FloatImageType::Pointer ImageMean = FloatImageType::New();
	ImageMean->SetRegions(data->GetRequestedRegion());
	ImageMean->Allocate();

	FloatImageType::Pointer ImageStd = FloatImageType::New();
	ImageStd->SetRegions(data->GetRequestedRegion());
	ImageStd->Allocate();

	FloatImageType::Pointer ImageSkw = FloatImageType::New();
	ImageSkw->SetRegions(data->GetRequestedRegion());
	ImageSkw->Allocate();

	ImageIterator it1(ImageMean,ImageMean->GetRequestedRegion());
	ImageIterator it2(ImageStd,ImageStd->GetRequestedRegion());
	ImageIterator it3(ImageSkw,ImageSkw->GetRequestedRegion());

	float mean, std, skw;

	for (it.GoToBegin(), it1.GoToBegin(), it2.GoToBegin(), it3.GoToBegin(); !it.IsAtEnd(); ++it, ++it1, ++it2, ++it3)
	{
		//Mean
		float sum = 0.0;
		for(unsigned int i = 0; i<it.Size(); ++i)
			sum += it.GetPixel(i);
		mean = sum/float(it.Size());

		//Std. Deviation
		sum = 0.0;
		for(unsigned int i = 0; i<it.Size(); ++i)
			sum += pow(it.GetPixel(i) - mean,2);
		std = sqrt(sum/float(it.Size()));

		//Skewness
		sum = 0.0;
		for(unsigned int i = 0; i<it.Size(); ++i)
			sum += pow((it.GetPixel(i) - mean)/std,3);
		skw = sum/float(it.Size());

		it1.Set(mean);
		it2.Set(std);
		it3.Set(skw);
	}

	WriterType::Pointer writer = WriterType::New();

	char savefile[256];

	writer->SetInput(ImageMean);
	sprintf(savefile, "%smean_%d_%s",argv[3],r,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();

	writer->SetInput(ImageStd);
	sprintf(savefile, "%sstd_%d_%s",argv[3],r,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();

	writer->SetInput(ImageSkw);
	sprintf(savefile, "%sskw_%d_%s",argv[3],r,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();
}

int main(int argc, char** argv)
{
	if(argc != 4)
	{
		std::cout << "Invalid Command! Usage: NeighborhoodFeatures <FileDirectory/> <InputFileName> <OutputDirectory/>" << std::endl;
		return -1;
	}

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(std::string(argv[1]+std::string(argv[2])));
	reader->Update();
	ImageType::Pointer data = ImageType::New();
	data = reader->GetOutput();

	ExtractFeatures(data, 1, argv);
	ExtractFeatures(data, 3, argv);

	return 0;
}