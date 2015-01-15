/*
Extract and save neighborhood features for mean, standard deviation and skewness of an image.
Two sets of 3 images with radius 1 and 3. (3x3x3 region and 7x7x7 region)
Input:  
Output:

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
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIterator;
typedef itk::ImageRegionIterator<ImageType> ImageIterator;

void ExtractFeatures(ImageType::Pointer data, int r, char* filename, char* savepath)
{

	std::cout << filename;

	//Extract Neighborhood features
	NeighborhoodIterator::RadiusType radius;
	radius.Fill(r);
	NeighborhoodIterator it(radius, data, data->GetRequestedRegion());
	
	ImageType::Pointer ImageMean = ImageType::New();
	ImageMean->SetRegions(data->GetRequestedRegion());
	ImageMean->Allocate();

	ImageType::Pointer ImageStd = ImageType::New();
	ImageStd->SetRegions(data->GetRequestedRegion());
	ImageStd->Allocate();

	ImageType::Pointer ImageSkw = ImageType::New();
	ImageSkw->SetRegions(data->GetRequestedRegion());
	ImageSkw->Allocate();

	ImageIterator out1(ImageMean,ImageMean->GetRequestedRegion());
	ImageIterator out2(ImageStd,ImageStd->GetRequestedRegion());
	ImageIterator out3(ImageSkw,ImageSkw->GetRequestedRegion());

	float mean, std, skw;

	for (it.GoToBegin(), out1.GoToBegin(), out2.GoToBegin(), out3.GoToBegin(); !it.IsAtEnd(); ++it, ++out1, ++out2, ++out3)
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

		out1.Set(mean);
		out2.Set(std);
		out3.Set(skw);
	}
	
	WriterType::Pointer writer = WriterType::New();

	char savefile[256];

	writer->SetInput(ImageMean);
	sprintf(savefile, "%smean_%d_%s",savepath,r,filename);
	writer->SetFileName(savefile);
	writer->Update();

	writer->SetInput(ImageStd);
	sprintf(savefile, "%sstd_%d_%s",savepath,r,filename);
	writer->SetFileName(savefile);
	writer->Update();

	writer->SetInput(ImageSkw);
	sprintf(savefile, "%sskw_%d_%s",savepath,r,filename);
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

	std::string filepath(argv[1]);
	std::string filename(argv[2]); 

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(strcat(argv[1],argv[2]));
	reader->Update();
	ImageType::Pointer data = ImageType::New();
	data = reader->GetOutput();

	ExtractFeatures(data, 1, argv[2], argv[3]);
	ExtractFeatures(data, 3, argv[2], argv[3]);

	return 0;
}