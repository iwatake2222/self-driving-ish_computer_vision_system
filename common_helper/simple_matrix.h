/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef SIMPLE_MATRIX_
#define SIMPLE_MATRIX_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>

class SimpleMatrix
{
public:
	SimpleMatrix()
	{
		rows = 0;
		cols = 0;
		data_array = std::vector<double>(0);
	}

	SimpleMatrix(int32_t _rows, int32_t _cols)
	{
		rows = _rows;
		cols = _cols;
		data_array = std::vector<double>(rows * cols);
	}

	SimpleMatrix(int32_t _rows, int32_t _cols, std::vector<double> _data_array)
	{
		rows = _rows;
		cols = _cols;
		data_array = _data_array;
		if (!CheckShape()) {
			throw std::out_of_range("Indata_arrayid shape at constructor");
		}
	}

	~SimpleMatrix() {}

	double& operator() (int32_t y, int32_t x)
	{
		if (y >= rows || x >= cols) {
			throw std::out_of_range("Invalid index");
		}
		return data_array[y * cols + x];
	}

	const double& operator() (int32_t y, int32_t x) const
	{
		if (y >= rows || x >= cols) {
			throw std::out_of_range("Invalid index");
		}
		return data_array[y * cols + x];
	}

	const SimpleMatrix operator+ (const SimpleMatrix& mat2) const
	{
		if (!CheckShape() || !mat2.CheckShape() || !CheckShapeSame(mat2)) {
			throw std::out_of_range("Invalid shape at add");
		}

		const SimpleMatrix& mat1 = *this;
		SimpleMatrix ret(mat1.rows, mat1.cols);

		for (int32_t y = 0; y < rows; y++) {
			for (int32_t x = 0; x < cols; x++) {
				ret(y, x) = mat1(y, x) + mat2(y, x);
			}
		}
		return ret;
	}

	const SimpleMatrix operator- (const SimpleMatrix& mat2) const
	{
		if (!CheckShape() || !mat2.CheckShape() || !CheckShapeSame(mat2)) {
			throw std::out_of_range("Invalid shape at sub");
		}

		const SimpleMatrix& mat1 = *this;
		SimpleMatrix ret(mat1.rows, mat1.cols);

		for (int32_t y = 0; y < rows; y++) {
			for (int32_t x = 0; x < cols; x++) {
				ret(y, x) = mat1(y, x) - mat2(y, x);
			}
		}
		return ret;
	}

	const SimpleMatrix operator* (const SimpleMatrix& mat2) const 
	{
		if (!CheckShape() || !mat2.CheckShape() || !CheckShapeMul(mat2)) {
			throw std::out_of_range("Invalid shape at mul");
		}
		
		const SimpleMatrix& mat1 = *this;
		SimpleMatrix ret(mat1.rows, mat2.cols);

		for (int32_t y = 0; y < mat1.rows; y++) {
			for (int32_t x = 0; x < mat2.cols; x++) {
				double sum = 0;
				for (int32_t i = 0; i < mat1.cols; i++) {
					sum += mat1(y, i) * mat2(i, x);
				}
				ret(y, x) = sum;
			}
		}
		return ret;
	}

	const SimpleMatrix operator* (const double& k) const
	{
		const SimpleMatrix& mat1 = *this;
		SimpleMatrix ret(mat1.rows, mat1.cols);

		for (int32_t y = 0; y < rows; y++) {
			for (int32_t x = 0; x < cols; x++) {
				ret(y, x) = mat1(y, x) * k;
			}
		}
		return ret;
	}

	SimpleMatrix Transpose() const
	{
		if (!CheckShape()) {
			throw std::out_of_range("Invalid shape at transpose");
		}
		
		const SimpleMatrix& mat1 = *this;
		SimpleMatrix ret(mat1.cols, mat1.rows);

		for (int32_t y = 0; y < rows; y++) {
			for (int32_t x = 0; x < cols; x++) {
				ret(x, y) = mat1(y, x);
			}
		}
		return ret;
	}

	SimpleMatrix Inverse() const
	{
		if (!CheckShape() || rows  != cols) {
			printf("Invalid shape at Inverse\n");
			throw std::out_of_range("SimpleMatrix");
		}

		SimpleMatrix mat = *this;
		int32_t n = mat.rows;
		SimpleMatrix I = SimpleMatrix::IdentityMatrix(n);
		
		for (int32_t y = 0; y < n; y++) {
			if (mat(y, y) == 0) {
				throw std::out_of_range("Tried to calculate an inverse of non - singular matrix");
			}
			double scale_to_1 = 1.0 / mat(y, y);
			for (int32_t x = 0; x < n; x++) {
				mat(y, x) *= scale_to_1;
				I(y, x) *= scale_to_1;
			}
			for (int32_t yy = 0; yy < n; yy++) {
				if (yy != y) {
					double scale_to_0 = mat(yy, y);
					for (int32_t x = 0; x < n; x++) {
						mat(yy, x) -= mat(y, x) * scale_to_0;
						I(yy, x) -= I(y, x) * scale_to_0;
					}
				}
			}
		}

		return I;
	}


	void Display() const
	{
		if (!CheckShape()) {
			throw std::out_of_range("Invalid shape at displays");
		}

		for (int32_t y = 0; y < rows; y++) {
			for (int32_t x = 0; x < cols; x++) {
				printf("%f ", (*this)(y, x));
			}
			printf("\n");
		}
	}

	static SimpleMatrix IdentityMatrix(int32_t n)
	{
		SimpleMatrix ret(n, n);
		for (int32_t i = 0; i < n; i++) {
			ret(i, i) = 1;
		}
		return ret;
	}

	static void Test()
	{
		try {
			SimpleMatrix mat1(2, 3, { 1, 2, 3, 4, 5, 6 });
			SimpleMatrix mat2(2, 3, { 7, 8, 9, 10, 11, 12 });
			SimpleMatrix mat3(3, 2, { 1, 2, 3, 4, 5, 6 });
			SimpleMatrix mat4(2, 2, { 1, 2, 3, 4 });
			SimpleMatrix mat5(3, 3, { 2, 2, 3, 4, 5, 6, 7, 8, 9 });
			SimpleMatrix mat6(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

			printf("\n--- mat1 ---\n");
			mat1.Display();
			
			printf("\n--- mat2 ---\n");
			mat2.Display();

			printf("\n--- add ---\n");
			SimpleMatrix matAdd = mat1 + mat2;
			matAdd.Display();

			printf("\n--- sub ---\n");
			SimpleMatrix matSub = mat1 - mat2;
			matSub.Display();

			printf("\n--- mul ---\n");
			SimpleMatrix matMul = mat1 * mat3;
			matMul.Display();

			printf("\n--- transpose ---\n");
			SimpleMatrix matTranspose = mat1.Transpose();
			matTranspose.Display();

			printf("\n--- Identity matrix ---\n");
			SimpleMatrix matI = SimpleMatrix::IdentityMatrix(3);
			matI.Display();

			printf("\n--- Inverse matrix 2x2 ---\n");
			SimpleMatrix matInv = mat4.Inverse();
			matInv.Display();
			(mat4 * matInv).Display();
			(matInv * mat4).Display();

			printf("\n--- Inverse matrix 3x3 ---\n");
			matInv = mat5.Inverse();
			matInv.Display();
			(mat5 * matInv).Display();
			(matInv * mat5).Display();

			printf("\n--- Inverse of non-singular matrix 3x3 ---\n");
			matInv = mat6.Inverse();
		} catch (std::exception& e) {
			printf("Exception: %s\n", e.what());
		}
	}

	bool CheckShape() const
	{
		if (static_cast<int32_t>(data_array.size()) != cols * rows) return false;
		return true;
	}

	bool CheckShapeSame(const SimpleMatrix& mat2) const
	{
		if (cols != mat2.cols) return false;
		if (rows != mat2.rows) return false;
		return true;
	}

	bool CheckShapeMul(const SimpleMatrix& mat2) const
	{
		if (cols != mat2.rows) return false;
		return true;
	}

	std::vector<double> data_array;
	int32_t rows;
	int32_t cols;
};


#endif
