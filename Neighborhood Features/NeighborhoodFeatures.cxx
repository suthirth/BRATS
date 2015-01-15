#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNeighborhoodIterator.h"

int main()
{
	typedef itk::Image<unsigned short, 3> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;
	typedef itk::ImageFileWriter<ImageType> WriterType;
	typedef itk::NeighborhoodIterator<ImageType> NeighborhoodIterator;
	typedef itk::ImageRegionIterator<ImageType> ImageIterator;

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	WriterType::Pointer writer = WriterType::New();
	reader->SetFileName("/home/suthirth/Dataset/1.mha");
	ImageType::Pointer data = ImageType::New();
	data = reader->GetOutput();

	//Neighborhood features of radius = 1
	ImageType::Pointer output = ImageType::New();
	NeighborhoodIterator::RadiusType radius;
	for (unsigned int i = 0; i < ImageType::ImageDimension; ++i)
		radius[i] = 1;

	NeighborhoodIterator it(radius, data, data->GetRequestedRegion());
	ImageIterator out(output,output->GetRequestedRegion());
	it.Begin();

	for (out = out.Begin(); !it.IsAtEnd(); ++it, ++out)
	{
		float sum = 0.0;
		for(unsigned int i = 0; i<it.Size(); ++i)
			sum += it.GetPixel(i);
		out.Set(sum/float(it.Size()));
	}

	writer->SetInput(output);
	writer->SetFileName("/home/suthirth/Dataset/mean.mha");
	writer->Update();

	return 0;
}