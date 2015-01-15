#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include <stdio.h>
#include <math.h>

int main()
{
	typedef itk::Image<unsigned short, 3> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;
	typedef itk::ImageFileWriter<ImageType> WriterType;
	typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIterator;
	typedef itk::ImageRegionIterator<ImageType> ImageIterator;

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	WriterType::Pointer writer = WriterType::New();
	reader->SetFileName("/home/suthirth/Dataset/1.mha");
	reader->Update();
	ImageType::Pointer data = ImageType::New();
	data = reader->GetOutput();

	//Neighborhood features of radius = 1
	NeighborhoodIterator::RadiusType radius;
	radius.Fill(1);
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
	
	writer->SetInput(ImageMean);
	writer->SetFileName("/home/suthirth/Dataset/mean.mha");
	writer->Update();

	writer->SetInput(ImageStd);
	writer->SetFileName("/home/suthirth/Dataset/std.mha");
	writer->Update();

	writer->SetInput(ImageSkw);
	writer->SetFileName("/home/suthirth/Dataset/skw.mha");
	writer->Update();

	return 0;
}