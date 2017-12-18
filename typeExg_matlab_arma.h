#ifndef TYPEEXG_MATLAB_ARMA_H
#define TYPEEXG_MATLAB_ARMA_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

/*
Possible mxClassID values and Types (and the mapping between them):
mxINT8_CLASS <==> char, byte
mxUINT8_CLASS <==> unsigned char, byte [used often]
mxINT16_CLASS <==> short
mxUINT16_CLASS <==> unsigned short
mxINT32_CLASS <==> int [used often]
mxUINT32_CLASS <==> unsigned int [used often]
mxINT64_CLASS <==> long long
mxUINT64_CLASS <==> unsigned long long
mxSINGLE_CLASS <==> float [used often]
mxDOUBLE_CLASS <==> double [used often]
Note: "uword" is the typedef for "unsigned int" and is used for used for matrix indices as well as all internal counters and loops
Note: "sword" is a typedef for a signed integer type


Demo of the functions.

========== Code =============

MatlabEngWrapper mew;
mew.init();
mew.exec("clear all; X = [1,2;3,4]; X = int32(X);");
mxArray* X = mew.receive("X");
arma::Mat<int> X_cv; matlab2arma<int>(X, X_cv, true); mxDestroyArray(X);
arma::Mat<int> Y_cv = X_cv + 2;
mxArray* Y; arma2matlab<int>(Y_cv, Y);
mew.send("Y", Y);

========== Output in Matlab =============

» whos, X, Y
Name      Size            Bytes  Class    Attributes

X         2x2                16  int32
Y         2x2                16  int32


X =

1           2
3           4


Y =

3           4
5           6


========== Code =============

MatlabEngWrapper mew;
mew.init();
mew.exec("clear all; X(:,:,1) = [1,2;3,4]; X(:,:,2) = [1,2;3,4]; X = single(X);");
mxArray* X = mew.receive("X");
arma::Cube<float> X_cv; matlab2arma<float>(X, X_cv, true); mxDestroyArray(X);
arma::Cube<float> Y_cv = X_cv + 2;
mxArray* Y; arma2matlab<float>(Y_cv, Y);
mew.send("Y", Y);

========== Output in Matlab =============

» whos, X, Y
Name      Size             Bytes  Class     Attributes

X         2x2x2               32  single
Y         2x2x2               32  single


X(:,:,1) =

1     2
3     4


X(:,:,2) =

1     2
3     4


Y(:,:,1) =

3     4
5     6


Y(:,:,2) =

3     4
5     6


*/

#include <armadillo>
#include "mex.h"
#include <cstring> // for memcpy


// this namespace contains helper functions to be used only in this file (not be called from outside)
// they are all put in a namespace to avoid clashing (resulting in linker errors) with other same
// function names in other header files
namespace hpers_TEMatArma
{
	// Template for mapping C primitive types to MATLAB types (commented out ones are not supported in Armadillo)
	template<class T> inline mxClassID getMatlabType() { return mxUNKNOWN_CLASS; }
	template<> inline mxClassID getMatlabType<char>()				{ return mxINT8_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned char>()		{ return mxUINT8_CLASS; }
	template<> inline mxClassID getMatlabType<short>()				{ return mxINT16_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned short>()		{ return mxUINT16_CLASS; }
	template<> inline mxClassID getMatlabType<int>()				{ return mxINT32_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned int>()		{ return mxUINT32_CLASS; }
	//template<> inline mxClassID getMatlabType<long long>()			{ return mxINT64_CLASS; }
	//template<> inline mxClassID getMatlabType<unsigned long long>() { return mxUINT64_CLASS; }
	template<> inline mxClassID getMatlabType<float>()				{ return mxSINGLE_CLASS; }
	template<> inline mxClassID getMatlabType<double>()				{ return mxDOUBLE_CLASS; }
}


// works for 2D matrices (real numbers, not complex)
template <typename T>
void arma2matlab(const arma::Mat<T>& matIn, mxArray* &matOut)
{
	int nrows = matIn.n_rows;
	int ncols = matIn.n_cols;
	
	matOut = mxCreateNumericMatrix(nrows, ncols, hpers_TEMatArma::getMatlabType<T>(), mxREAL);
	T *dst_pointer = (T*)mxGetData(matOut);
	
	unsigned long count = 0;
	
	for(int j=0; j<ncols; j++)
		for(int i=0; i<nrows; i++)
			dst_pointer[count++] = matIn.at(i,j);
}

// works for 3D matrices (real numbers, not complex)
// if arma Cube has number of slices = 1, resulting mxArray matOut will automatically have 2 dimensions
template <typename T>
void arma2matlab(const arma::Cube<T>& matIn, mxArray* &matOut)
{	
	int nrows = matIn.n_rows;
	int ncols = matIn.n_cols;
	int nchannels = matIn.n_slices;
	
	const mwSize ndims = 3;
	mwSize dims[ndims] = { nrows, ncols, nchannels };
	matOut = mxCreateNumericArray(ndims, dims, hpers_TEMatArma::getMatlabType<T>(), mxREAL);
	T *dst_pointer = (T*)mxGetData(matOut);
	
	unsigned long count = 0;
	
	for(int k=0; k<nchannels; k++)
		for(int j=0; j<ncols; j++)
			for(int i=0; i<nrows; i++)
				dst_pointer[count++] = matIn.at(i,j,k);
}

// works for 2D matrices (real numbers, not complex)
template <typename T>
void matlab2arma(mxArray* matIn, arma::Mat<T> &matOut, bool copy_aux_mem = false, bool strict = false)
{
	matOut = arma::Mat<T>((T*)mxGetData(matIn), mxGetM(matIn), mxGetN(matIn), copy_aux_mem, strict);
}

// works for 3D matrices (real numbers, not complex)
// this also works for 2D matrices; in this case, arma:Cube output will have number of slices = 1
template <typename T>
void matlab2arma(mxArray* matIn, arma::Cube<T> &matOut, bool copy_aux_mem = false, bool strict = false)
{
	int ndims = (int)mxGetNumberOfDimensions(matIn);
	const size_t *dims = mxGetDimensions(matIn);
	unsigned int nrows = (unsigned int) dims[0];
	unsigned int ncols = (unsigned int) dims[1];
	unsigned int nchannels = ndims == 2 ? 1 : (unsigned int)dims[2];

	matOut = arma::Cube<T>((T*)mxGetData(matIn), nrows, ncols, nchannels, copy_aux_mem, strict);
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// Special faster functions for arma matrix inputs which are guaranteed to be stored
// contiguously. Faster speed comes from using std:memcpy instead of looping
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

// works for 2D matrices (real numbers, not complex)
// actually this function is dangerous in that it assumes that
// the input matIn is stored contiguously which is the case for almost
// all arma matrices except for the ones obtained using non-contiguous views
// by using non-continuous indexing. Therefore, unless I'm sure that
// matIn is not a non-contiguous matrix, don't use this function.
template <typename T>
void arma2matlabContig(const arma::Mat<T>& matIn, mxArray* &matOut)
{
	matOut = mxCreateNumericMatrix(matIn.n_rows, matIn.n_cols, hpers_TEMatArma::getMatlabType<T>(), mxREAL);
	T *dst_pointer = (T*)mxGetData(matOut);
	const T *src_pointer = (T*)matIn.memptr();
	std::memcpy(dst_pointer, src_pointer, sizeof(T)*matIn.n_elem);
}

// works for 3D matrices (real numbers, not complex)
// if arma Cube has number of slices = 1, resulting mxArray matOut will automatically have 2 dimensions
// actually this function is dangerous in that it assumes that
// the input matIn is stored contiguously which is the case for almost
// all arma matrices except for the ones obtained using non-contiguous views
// by using non-continuous indexing. Therefore, unless I'm sure that
// matIn is not a non-contiguous matrix, don't use this function.
template <typename T>
void arma2matlabContig(const arma::Cube<T>& matIn, mxArray* &matOut)
{	
	const mwSize ndims = 3;
	mwSize dims[ndims] = { matIn.n_rows, matIn.n_cols, matIn.n_slices };
	matOut = mxCreateNumericArray(ndims, dims, hpers_TEMatArma::getMatlabType<T>(), mxREAL);
	T *dst_pointer = (T*)mxGetData(matOut);
	const T *src_pointer = (T*)matIn.memptr();
	std::memcpy(dst_pointer, src_pointer, sizeof(T)*matIn.n_elem);
}


#endif