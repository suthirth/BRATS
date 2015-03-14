#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>


typedef itk::Image<float, 3> ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageRegionIterator<ImageType> ImIterator;

int main(int argc, char** argv)
{
	if(argc < 4)
	{
		std::cerr << "Usage" <<std::endl;
		std::cerr << argv[0] << " <InputFile> <LabelFile> <CSF Label Value>" << std::endl;
		return -1;
	}

	//Read file
	ImageType::Pointer data = ImageType::New();
	ImageType::Pointer label = ImageType::New();	
	ReaderType::Pointer reader = ReaderType::New();
	ReaderType::Pointer reader2 = ReaderType::New();
	
	reader->SetFileName(argv[1]);
	reader->Update();
	data = reader->GetOutput();

	reader2->SetFileName(argv[2]);
	reader2->Update();
	label = reader2->GetOutput();

	ImIterator iit(data,data->GetRequestedRegion());
   	ImIterator lit(label,label->GetRequestedRegion());

   	float sum = 0.0;
   	int count = 0;
   	float avg;
   	int csf_l = atoi(argv[3]);

   	for(iit.GoToBegin(),lit.GoToBegin();!iit.IsAtEnd();++iit,++lit)
  		if(lit.Get() == csf_l)
        {
        	sum+= float(iit.Get());
        	count++;
        }	
    
    if (count ==0)
    	std::cerr << "No voxels with given label found!" << std::endl;
    else
    {	
    	avg = sum/float(count);
    	std::cout << avg << std::endl;
	}

	return 0;
}