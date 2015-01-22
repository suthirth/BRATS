/*
Extract all features for given MRI 3D image.
Two sets of neighborhood features with radius 1 and 3. (3x3x3 region and 7x7x7 region)
Two images with Gaussian filters of sigma = 3 and 7 
Input: 3D File in ITK readable format.  
Usage: ExtractFeatures <FileDirectory/> <InputFileName> <OutputDirectory/>
Output: Files are saved in <OutputDirectory> with prefix.
*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include "itkDiscreteGaussianImageFilter.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef itk::Image<unsigned short, 3> ImageType;
typedef itk::Image<float, 3> FloatImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<FloatImageType> WriterType;
typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIterator;
typedef itk::ImageRegionIterator<FloatImageType> ImageIterator;
typedef itk::DiscreteGaussianImageFilter<ImageType, FloatImageType> GaussianFilterType;

void NeighborhoodFeatures(ImageType::Pointer data, int r, char* argv[])
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

	FloatImageType::Pointer ImageKurt = FloatImageType::New();
	ImageKurt->SetRegions(data->GetRequestedRegion());
	ImageKurt->Allocate();

	FloatImageType::Pointer ImageMax = FloatImageType::New();
	ImageMax->SetRegions(data->GetRequestedRegion());
	ImageMax->Allocate();

	FloatImageType::Pointer ImageMin = FloatImageType::New();
	ImageMin->SetRegions(data->GetRequestedRegion());
	ImageMin->Allocate();

	ImageIterator it1(ImageMean,ImageMean->GetRequestedRegion());
	ImageIterator it2(ImageStd,ImageStd->GetRequestedRegion());
	ImageIterator it3(ImageSkw,ImageSkw->GetRequestedRegion());
	ImageIterator it4(ImageKurt,ImageKurt->GetRequestedRegion());
	ImageIterator it5(ImageMax,ImageMax->GetRequestedRegion());
	ImageIterator it6(ImageMin,ImageMin->GetRequestedRegion());
	
	for (it.GoToBegin(), it1.GoToBegin(), it2.GoToBegin(), it3.GoToBegin(), it4.GoToBegin(), it5.GoToBegin(), it6.GoToBegin(); !it.IsAtEnd(); ++it, ++it1, ++it2, ++it3, ++it4, ++it5, ++it6)
	{
		float mean, std, skw, kurt;

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
		float accum = 0.0;
		for(unsigned int i = 0; i<it.Size(); ++i)
		{
			sum += pow((it.GetPixel(i) - mean)/std,3);
			accum += pow((it.GetPixel(i) - mean)/std,4);
		}	
		skw = sum/float(it.Size());
		kurt = accum/float(it.Size());

		float max = it.GetPixel(0);
		float min = it.GetPixel(0);
		for(unsigned int i = 0; i<it.Size(); ++i)
		{
			if(it.GetPixel(i) > max)
				max = it.GetPixel(i);
			if(it.GetPixel(i) < min)
				min = it.GetPixel(i);
		}

		it1.Set(mean);
		it2.Set(std);
		it3.Set(skw);
		it4.Set(kurt);
		it5.Set(max);
		it6.Set(min);

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

	writer->SetInput(ImageKurt);
	sprintf(savefile, "%skurt_%d_%s",argv[3],r,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();

	writer->SetInput(ImageMax);
	sprintf(savefile, "%smax_%d_%s",argv[3],r,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();

	writer->SetInput(ImageMin);
	sprintf(savefile, "%smin_%d_%s",argv[3],r,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();

}

int main(int argc, char** argv)
{
	if(argc != 4)
	{
		std::cerr << "Usage" <<std::endl;
		std::cerr << argv[0] << " <FileDirectory/> <InputFileName> <OutputDirectory/>" << std::endl;
		return -1;
	}

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(std::string(argv[1]+std::string(argv[2])));
	reader->Update();
	ImageType::Pointer data = ImageType::New();
	data = reader->GetOutput();

	NeighborhoodFeatures(data, 1, argv);
	NeighborhoodFeatures(data, 3, argv);

	GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
  	gaussianFilter->SetInput(data);
  	gaussianFilter->SetVariance(3);

	WriterType::Pointer writer = WriterType::New();
	char savefile[256];

	writer->SetInput(gaussianFilter->GetOutput());
	sprintf(savefile, "%sgauss_%d_%s",argv[3],3,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();  	

	gaussianFilter->SetVariance(7);
	writer->SetInput(gaussianFilter->GetOutput());
	sprintf(savefile, "%sgauss_%d_%s",argv[3],7,argv[2]);
	writer->SetFileName(savefile);
	writer->Update();  	

	return 0;
}