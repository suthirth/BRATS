/*
Normalization
*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkHistogramMatchingImageFilter.h"
#include <stdio.h>


typedef float PixelType;
typedef itk::Image<PixelType, 3> ImageType;
typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> MatchingFilterType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;

int main(int argc, char** argv)
{

	if(argc < 4)
	{
		std::cout << "Invalid Command! Usage: Normalization <Output File> <Source File> <Ref File> {No. of Bins} {Points}" << std::endl;
		return -1;
	}

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	WriterType::Pointer writer = WriterType::New();
	
	std::cout << argv[2] << std::endl;
	reader->SetFileName(argv[2]);
	reader->Update();

	ImageType::Pointer source = ImageType::New();
	source = reader->GetOutput();

	ReaderType::Pointer reader2 = ReaderType::New();
	reader2->SetFileName(argv[3]);
	reader2->Update();

	ImageType::Pointer reference = ImageType::New();
	reference = reader2->GetOutput();
	
	long bins = 255;
	if (argc >= 5)
   		bins = atoi(argv[4]);

	long points = 64;
	if (argc >= 6)
   		points = atoi(argv[5]);
	
	typename MatchingFilterType::Pointer match = MatchingFilterType::New();
	match->SetSourceImage(source);
	match->SetReferenceImage(reference);
	match->SetNumberOfHistogramLevels(bins);
	match->SetNumberOfMatchPoints(points); 	
	match->ThresholdAtMeanIntensityOn();
	match->Update();

	ImageType::Pointer output = ImageType::New();
	output = match->GetOutput();

	writer->SetInput(output);
	writer->SetFileName(argv[1]);
	writer->Update();

	return 0;
}

