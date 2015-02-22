/*
Extract all features for given MRI 3D image.
Two sets of neighborhood features with radius 1 and 3. (3x3x3 region and 7x7x7 region)
Two images with Gaussian filters of sigma = 3 and 7 
Input: 3D File in ITK readable format.  
Usage: ExtractFeatures <FileDirectory/> <InputFileName> <OutputDirectory/>
Output: Files are saved in <OutputDirectory> with prefix.
*/

//TODO: Multi-threading enable

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include "itkDiscreteGaussianImageFilter.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef itk::Image<float, 3> ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIterator;
typedef itk::ImageRegionIterator<ImageType> ImageIterator;
typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussianFilterType;

ImageType::Pointer MakeImage(ImageType::Pointer Im)
{
	ImageType::Pointer NewIm = ImageType::New();
	NewIm->SetRegions(Im->GetRequestedRegion());
	NewIm->Allocate();
	return NewIm;
}

void NeighborhoodFeatures(ImageType::Pointer data, int r, char* f)
{
	char* idx[6] = {"mean","std","skw","kurt","max","min"};

	//Extract Neighborhood features
	NeighborhoodIterator::RadiusType radius;
	radius.Fill(r);
	NeighborhoodIterator it(radius, data, data->GetRequestedRegion());

	WriterType::Pointer writer = WriterType::New();
	
	for (int stat=1; stat<=6; stat++)
	{
		ImageType::Pointer Im = MakeImage(data);
		ImageIterator it1(Im, Im->GetRequestedRegion());	
		float sum = 0.0;
		float mean, max, min;

		for (it.GoToBegin(), it1.GoToBegin();!it.IsAtEnd();++it, ++it1)
		{
			switch (stat)
			{
				case 1: //Mean
					for (int i=0; i<it.Size(); ++i)
						sum += it.GetPixel(i);
					it1.Set(sum/float(it.Size()));
					break;

				case 2: //Std
					for (int i=0; i<it.Size(); ++i)
						sum += it.GetPixel(i);
					mean = sum/float(it.Size());

					for(int i = 0; i<it.Size(); ++i)
						sum += pow(it.GetPixel(i) - mean,2);
					it1.Set(sqrt(sum/float(it.Size())));
					break;

				case 3: // Skewness
					for (int i=0; i<it.Size(); ++i)
						sum += it.GetPixel(i);
					mean = sum/float(it.Size());
					
					for(int i = 0; i<it.Size(); ++i)
						sum += pow((it.GetPixel(i) - mean)/std,3);				
					it1.Set(sum/float(it.Size()));
					break;
					
				case 4: // Kurtosis
					for (int i=0; i<it.Size(); ++i)
						sum += it.GetPixel(i);
					mean = sum/float(it.Size());
					
					for(int i = 0; i<it.Size(); ++i)
						sum += pow((it.GetPixel(i) - mean)/std,4);				
					it1.Set(sum/float(it.Size()));
					break;
					
				case 5: //Max
					max = it.GetPixel(0);
					for(int i = 0; i<it.Size(); ++i)
						if(it.GetPixel(i) > max)
							max = it.GetPixel(i);		
					it1.Set(max);
					break;
					
				case 6: //Min
					min = it.GetPixel(0);
					for(int i = 0; i<it.Size(); ++i)
						if(it.GetPixel(i) < min)
							min = it.GetPixel(i);		
					it1.Set(min);
					break;
					
			}
		}

		char savefile[256];
		sprintf(savefile, "%s_%s_%d.nii.gz",f,idx[(stat-1)],r);
		
		writer->SetInput(Im);
		writer->SetFileName(savefile);
		writer->Update();	

	}
}

void Gauss(ImageType::Pointer data, int r, char* f)
{
	GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
  	gaussianFilter->SetInput(data);
  	gaussianFilter->SetVariance(r);

  	WriterType::Pointer writer = WriterType::New();
  	char savefile[256];
	sprintf(savefile, "%s_gauss_%d.nii.gz",f,r);

	writer->SetInput(gaussianFilter->GetOutput());
	writer->SetFileName(savefile);
	writer->Update();	

}

int main(int argc, char** argv)
{
	
	//Usage 
	if(argc != 6)
	{
		std::cerr << "Usage" <<std::endl;
		std::cerr << argv[0] << "<Output_Prefix> <T1_Image> <T1C_Image> <T2_Image> <FLAIR_Image>" << std::endl;
		return -1;
	}

	std::cout << "Extracting Features for " << argv[2] << argv[3] << argv[4] << argv[5] << std::endl;

	//Read files
	ReaderType::Pointer reader = ReaderType::New();
	ImageType::Pointer data = ImageType::New();

	//Extract sequence specific features
	for (int i = 2; i <= 5; i++)
	{
		reader->SetFileName(argv[i]);
		reader->Update();
		data = reader->GetOutput();
		NeighborhoodFeatures(data, 1, argv[i]);
		NeighborhoodFeatures(data, 3, argv[i]);
		Gauss(data, 3, argv[i]);
		Gauss(data, 7, argv[i]);
	}
	
	return 0;
}