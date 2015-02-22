/*
Extract all features for given MRI 3D image.
Two sets of neighborhood features with radius 1 and 3. (3x3x3 region and 7x7x7 region)
Two images with Gaussian filters of sigma = 3 and 7 
Input: 3D File in ITK readable format.  
Usage: ExtractFeatures <FileDirectory/> <InputFileName> <OutputDirectory/>
Output: Files are saved in <OutputDirectory> with prefix.
*/


//TODO: Multi-threading enable
// Make mask, check atropos

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef itk::Image<float, 3> ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::ImageRegionIterator<ImageType> ImIterator;

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		std::cerr << "Usage" <<std::endl;
		std::cerr << argv[0] << " <InputFile> <OutputPrefix>" << std::endl;
		return -1;
	}

	//Read file
	ImageType::Pointer data = ImageType::New();
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(argv[1]);
	reader->Update();
	data = reader->GetOutput();

	//Create Mask
	ImageType::Pointer mask = ImageType::New();
	mask->SetRegions(data->GetRequestedRegion());
	mask->Allocate();
	
	ImIterator mit(mask, mask->GetRequestedRegion()); 
  	ImIterator it(data,data->GetRequestedRegion());
   	
   	for(it.GoToBegin(),mit.GoToBegin();!it.IsAtEnd();++it, ++mit)
  	{
  		if(it.Get()==0)
        	mit.Set(0);
      	else
        	mit.Set(1);
  	}

  	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(mask);
	writer->SetFileName(argv[2]);
	writer->Update();  	

	std::cout << "Saving Mask for " << argv[1] << " at " << argv[2] << std::endl;

	return 0;
}