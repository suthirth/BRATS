/*
N4ITK Bias Correction 
Bias Field Correction for MRI Images using N4ITK
N4ITK: Nick’s N3 ITK Implementation For MRI Bias Field Correction (Nicholas J. Tustison and James C. Gee)

Input: 3D MRI Image
Usage: N4Bias <FileDirectory/> <InputFileName> <OutputDirectory/>
Output: Bias Field Corrected Image

*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include <stdio.h>

typedef itk::Image<unsigned short, 3> ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::N4BiasFieldCorrectionImageFilter<ImageType,ImageType, ImageType> N4FilterType;

int main(int argc, char** argv)
{

	if(argc != 4)
	{
		std::cout << "Invalid Command! Usage: N4Bias <FileDirectory/> <InputFileName> <OutputDirectory/>" << std::endl;
		return -1;
	}

	//Read file
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(std::string(argv[1]+std::string(argv[2])));
	reader->Update();
	ImageType::Pointer data = ImageType::New();
	data = reader->GetOutput();

	N4FilterType::Pointer n4filter = N4FilterType::New();
	n4filter->SetInput(data);
	n4filter->Update();

	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(n4filter->GetOutput());
	writer->SetFileName(std::string(argv[3])+std::string("n4_")+std::string(argv[2]));
	writer->Update();

	return 0;
}