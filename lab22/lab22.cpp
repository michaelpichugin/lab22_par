#include <iostream>
#include <fstream>
#include <string.h>
#include <omp.h>

// Определение типа для функций тестирования (имеют общуу структуру но вызывают соответствующие функции)
typedef double(*TestFunctTempl)(double** A, double** B, double** C, int& M, int& N, int& K, int& linearMultBlockSize);
// Функция рассчета среднеарифметического значения в доверительном интервале
double AvgTrustedInterval(double& avg, double*& times, int& cnt);
// Функция для замера времени Функции работы с данными 
double TestIter(void* Funct, int iterations, double** A, double** B, double** C, int M, int N, int K, int linearMultBlockSize);

/// ФУНКЦИИ РАБОТЫ С ДАННЫМИ

// Заполнение
double FillArr_Single(double** A, double** B, int M, int N, int K);
double FillArr_Par(double** A, double** B, int M, int N, int K);
//Перемножение
double MulArr_Single(double** A, double** B, double** C, int M, int N, int K);
double MulArr_Par1(double** A, double** B, double** C, int M, int N, int K);
double MulArr_Par2(double** A, double** B, double** C, int M, int N, int K);
double MulArr_Single_Fast(double** A, double** B, double** C, int M, int N, int K, int linearMultBlockSize);
double MulArr_Par_Fast(double** A, double** B, double** C, int M, int N, int K, int linearMultBlockSize);

/// ФУНКЦИИ ТЕСТИРОВАНИЯ

// Заполнение
double TestFillArr_Single(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return FillArr_Single(A, B, N, M, K);
}
double TestFillArr_Par(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return FillArr_Par(A, B, N, M, K);
}
// Перемножение
double TestMulArr_Single(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return MulArr_Single(A, B, C, N, M, K);
}
double TestMulArr_Par1(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return MulArr_Par1(A, B, C, N, M, K);
}
double TestMulArr_Par2(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return MulArr_Par2(A, B, C, N, M, K);
}
double TestMulArr_Single_Fast(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return MulArr_Single_Fast(A, B, C, N, M, K, linearMultBlockSize);
}
double TestMulArr_Par_Fast(double** A, double** B, double** C, int& N, int& M, int& K, int& linearMultBlockSize)
{
	return MulArr_Par_Fast(A, B, C, N, M, K, linearMultBlockSize);
}

/// РЕАЛИЗАЦИЯ ФУНКЦИЙ

// Заполнение
double FillArr_Single(double** A, double** B, int N, int M, int K)
{
	double t_start = omp_get_wtime();

	//Заполнение матрицы A
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			A[i][j] = cos(j) / sin(i + 1);
	//Заполнение матрицы B
	for (int i = 0; i < M; i++)
		for (int j = 0; j < K; j++)
			B[i][j] = cos(j) / sin(i + 1);

	double t_end = omp_get_wtime();
	return (t_end - t_start) * 1000;
}
double FillArr_Par(double** A, double** B, int M, int N, int K)
{
	double t_start = omp_get_wtime();

	//Заполнение матрицы A
#pragma omp parallel for
	for (int i = 0; i < N; i++)
#pragma omp parallel for
		for (int j = 0; j < M; j++)
			A[i][j] = cos(j) / sin(i + 1);

	//Заполнение матрицы B
#pragma omp parallel for
	for (int i = 0; i < M; i++)
#pragma omp parallel for
		for (int j = 0; j < K; j++)
			B[i][j] = cos(j) / sin(i + 1);

	double t_end = omp_get_wtime();
	return (t_end - t_start) * 1000;
}
// Перемножение
double MulArr_Single(double** A, double** B, double** C, int M, int N, int K)
{
	double time_start = omp_get_wtime();
	double** mtr = new double* [N];

	for (int i = 0; i < N; i++)
		mtr[i] = new double[M];

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			mtr[j][i] = B[i][j];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double tmp = 0;
			for (int k = 0; k < M; k++) {
				tmp += A[i][k] * mtr[j][k];
			}
			C[i][j] = tmp;
		}
	}

	for (int i = 0; i < N; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;


	double time_stop = omp_get_wtime();
	return (time_stop - time_start) * 1000;
}
//оптимизация путем хранения в локальной переменной элементов столбца
double MulArr_Par1(double** A, double** B, double** C, int M, int N, int K)
{
	double time_start = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < N; i++)
#pragma omp parallel for
		for (int j = 0; j < N; j++)
		{
			C[i][j] = 0;
		}
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
#pragma omp parallel for
		for (int k = 0; k < N; k++) {
			double tmpV = A[i][k];
#pragma omp parallel for
			for (int j = 0; j < M; j++) {
				C[i][j] += tmpV * B[k][j];
			}
		}
	}
	double time_stop = omp_get_wtime();
	return (time_stop - time_start) * 1000;
}
//использует механизма группировки данных в памяти путем транспонирования и локальную переменную
double MulArr_Par2(double** A, double** B, double** C, int M, int N, int K)
{
	double time_start = omp_get_wtime();
	double** mtr = new double* [N];
#pragma omp parallel for
	for (int i = 0; i < N; i++)
		mtr[i] = new double[M];

#pragma omp parallel for
	for (int i = 0; i < M; i++)
#pragma omp parallel for
		for (int j = 0; j < N; j++)
			mtr[j][i] = B[i][j];

#pragma omp parallel for
	for (int i = 0; i < N; i++) {
#pragma omp parallel for
		for (int j = 0; j < N; j++) {
			double tmp = 0;
#pragma omp parallel for
			for (int k = 0; k < M; k++) {
				tmp += A[i][k] * mtr[j][k];
			}
			C[i][j] = tmp;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;


	double time_stop = omp_get_wtime();
	return (time_stop - time_start) * 1000;
}

// Всё для Штрассена
// сумма
int ADD(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
		}
	}
	return 0;
}
// разность
int SUB(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
	for (int i = 0; i < MatrixSize; i++)
	{
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
		}
	}
	return 0;
}
// умножение матриц
int MUL(double** MatrixA, double** MatrixB, double** MatrixResult, int MatrixSize)
{
	double** mtr = new double* [MatrixSize];
	for (int i = 0; i < MatrixSize; i++)
		mtr[i] = new double[MatrixSize];
	for (int i = 0; i < MatrixSize; i++)
		for (int j = 0; j < MatrixSize; j++)
			mtr[j][i] = MatrixB[i][j];
	for (int i = 0; i < MatrixSize; i++) {
		for (int j = 0; j < MatrixSize; j++) {
			double tmp = 0;
			for (int k = 0; k < MatrixSize; k++) {
				tmp += MatrixA[i][k] * mtr[j][k];
			}
			MatrixResult[i][j] = tmp;
		}
	}
	for (int i = 0; i < MatrixSize; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;

	return 0;
}

int Strassen(double** MatrixA, double** MatrixB, double** MatrixC, int MatrixSize, int linearMultBlockSize)
{
	int HalfSize = MatrixSize / 2;

	if (MatrixSize <= linearMultBlockSize)
	{
		MUL(MatrixA, MatrixB, MatrixC, MatrixSize);
	}
	else
	{
		///////////// Выделение памяти под матрицы
		double** A11;
		double** A12;
		double** A21;
		double** A22;

		double** B11;
		double** B12;
		double** B21;
		double** B22;

		double** C11;
		double** C12;
		double** C21;
		double** C22;

		double** M1;
		double** M2;
		double** M3;
		double** M4;
		double** M5;
		double** M6;
		double** M7;
		double** AResult;
		double** BResult;


		A11 = new double* [HalfSize];
		A12 = new double* [HalfSize];
		A21 = new double* [HalfSize];
		A22 = new double* [HalfSize];

		B11 = new double* [HalfSize];
		B12 = new double* [HalfSize];
		B21 = new double* [HalfSize];
		B22 = new double* [HalfSize];

		C11 = new double* [HalfSize];
		C12 = new double* [HalfSize];
		C21 = new double* [HalfSize];
		C22 = new double* [HalfSize];

		M1 = new double* [HalfSize];
		M2 = new double* [HalfSize];
		M3 = new double* [HalfSize];
		M4 = new double* [HalfSize];
		M5 = new double* [HalfSize];
		M6 = new double* [HalfSize];
		M7 = new double* [HalfSize];

		AResult = new double* [HalfSize];
		BResult = new double* [HalfSize];


		for (int i = 0; i < HalfSize; i++)
		{
			A11[i] = new double[HalfSize];
			A12[i] = new double[HalfSize];
			A21[i] = new double[HalfSize];
			A22[i] = new double[HalfSize];

			B11[i] = new double[HalfSize];
			B12[i] = new double[HalfSize];
			B21[i] = new double[HalfSize];
			B22[i] = new double[HalfSize];

			C11[i] = new double[HalfSize];
			C12[i] = new double[HalfSize];
			C21[i] = new double[HalfSize];
			C22[i] = new double[HalfSize];

			M1[i] = new double[HalfSize];
			M2[i] = new double[HalfSize];
			M3[i] = new double[HalfSize];
			M4[i] = new double[HalfSize];
			M5[i] = new double[HalfSize];
			M6[i] = new double[HalfSize];
			M7[i] = new double[HalfSize];

			AResult[i] = new double[HalfSize];
			BResult[i] = new double[HalfSize];
		}
		/////////////////////////////////////////

				//Разделение матрицы на четыре части
		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				A11[i][j] = MatrixA[i][j];
				A12[i][j] = MatrixA[i][j + HalfSize];
				A21[i][j] = MatrixA[i + HalfSize][j];
				A22[i][j] = MatrixA[i + HalfSize][j + HalfSize];

				B11[i][j] = MatrixB[i][j];
				B12[i][j] = MatrixB[i][j + HalfSize];
				B21[i][j] = MatrixB[i + HalfSize][j];
				B22[i][j] = MatrixB[i + HalfSize][j + HalfSize];

			}
		}

		//P1 == M1[][]
		ADD(A11, A22, AResult, HalfSize);
		ADD(B11, B22, BResult, HalfSize);
		Strassen(AResult, BResult, M1, HalfSize, linearMultBlockSize);


		//P2 == M2[][]
		ADD(A21, A22, AResult, HalfSize);              //M2=(A21+A22)B11
		Strassen(AResult, B11, M2, HalfSize, linearMultBlockSize);       //Mul(AResult,B11,M2);

		//P3 == M3[][]
		SUB(B12, B22, BResult, HalfSize);              //M3=A11(B12-B22)
		Strassen(A11, BResult, M3, HalfSize, linearMultBlockSize);       //Mul(A11,BResult,M3);

		//P4 == M4[][]
		SUB(B21, B11, BResult, HalfSize);           //M4=A22(B21-B11)
		Strassen(A22, BResult, M4, HalfSize, linearMultBlockSize);       //Mul(A22,BResult,M4);

		//P5 == M5[][]
		ADD(A11, A12, AResult, HalfSize);           //M5=(A11+A12)B22
		Strassen(AResult, B22, M5, HalfSize, linearMultBlockSize);       //Mul(AResult,B22,M5);


		//P6 == M6[][]
		SUB(A21, A11, AResult, HalfSize);
		ADD(B11, B12, BResult, HalfSize);             //M6=(A21-A11)(B11+B12)
		Strassen(AResult, BResult, M6, HalfSize, linearMultBlockSize);    //Mul(AResult,BResult,M6);

		//P7 == M7[][]
		SUB(A12, A22, AResult, HalfSize);
		ADD(B21, B22, BResult, HalfSize);             //M7=(A12-A22)(B21+B22)
		Strassen(AResult, BResult, M7, HalfSize, linearMultBlockSize);     //Mul(AResult,BResult,M7);

		//C11 = M1 + M4 - M5 + M7;
		ADD(M1, M4, AResult, HalfSize);
		SUB(M7, M5, BResult, HalfSize);
		ADD(AResult, BResult, C11, HalfSize);

		//C12 = M3 + M5;
		ADD(M3, M5, C12, HalfSize);

		//C21 = M2 + M4;
		ADD(M2, M4, C21, HalfSize);

		//C22 = M1 + M3 - M2 + M6;
		ADD(M1, M3, AResult, HalfSize);
		SUB(M6, M2, BResult, HalfSize);
		ADD(AResult, BResult, C22, HalfSize);


		// Сбор частей в единую матрицу
		for (int i = 0; i < HalfSize; i++)
		{
			for (int j = 0; j < HalfSize; j++)
			{
				MatrixC[i][j] = C11[i][j];
				MatrixC[i][j + HalfSize] = C12[i][j];
				MatrixC[i + HalfSize][j] = C21[i][j];
				MatrixC[i + HalfSize][j + HalfSize] = C22[i][j];
			}
		}

		// очистка памяти
		for (int i = 0; i < HalfSize; i++)
		{
			delete[] A11[i]; delete[] A12[i]; delete[] A21[i]; 	delete[] A22[i];
			delete[] B11[i]; delete[] B12[i]; delete[] B21[i];	delete[] B22[i];
			delete[] C11[i]; delete[] C12[i]; delete[] C21[i];	delete[] C22[i];
			delete[] M1[i]; delete[] M2[i]; delete[] M3[i]; delete[] M4[i];
			delete[] M5[i]; delete[] M6[i]; delete[] M7[i];
			delete[] AResult[i]; delete[] BResult[i];
		}
		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
		delete[] M1; delete[] M2; delete[] M3; delete[] M4; delete[] M5;
		delete[] M6; delete[] M7;
		delete[] AResult;
		delete[] BResult;


	}
	return 0;
}

int ADDPar(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
#pragma omp parallel for
	for (int i = 0; i < MatrixSize; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
		}
	}
	return 0;
}

int SUBPar(double**& MatrixA, double**& MatrixB, double**& MatrixResult, int MatrixSize)
{
#pragma omp parallel for
	for (int i = 0; i < MatrixSize; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < MatrixSize; j++)
		{
			MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
		}
	}
	return 0;
}

int MULPar(double** MatrixA, double** MatrixB, double** MatrixResult, int MatrixSize)
{
	double** mtr = new double* [MatrixSize];
#pragma omp parallel for
	for (int i = 0; i < MatrixSize; i++)
		mtr[i] = new double[MatrixSize];
#pragma omp parallel for
	for (int i = 0; i < MatrixSize; i++)
#pragma omp parallel for
		for (int j = 0; j < MatrixSize; j++)
			mtr[j][i] = MatrixB[i][j];
#pragma omp parallel for
	for (int i = 0; i < MatrixSize; i++) {
#pragma omp parallel for
		for (int j = 0; j < MatrixSize; j++) {
			double tmp = 0;
#pragma omp parallel for
			for (int k = 0; k < MatrixSize; k++) {
				tmp += MatrixA[i][k] * mtr[j][k];
			}
			MatrixResult[i][j] = tmp;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < MatrixSize; i++)
	{
		delete mtr[i];
	}
	delete[] mtr;

	return 0;
}

int StrassenPar(double** MatrixA, double** MatrixB, double** MatrixC, int MatrixSize, int linearMultBlockSize)
{
	int HalfSize = MatrixSize / 2;

	if (MatrixSize <= linearMultBlockSize)
	{
		MULPar(MatrixA, MatrixB, MatrixC, MatrixSize);
	}
	else
	{
		///////////// Выделение памяти под матрицы
		double** A11;
		double** A12;
		double** A21;
		double** A22;

		double** B11;
		double** B12;
		double** B21;
		double** B22;

		double** C11;
		double** C12;
		double** C21;
		double** C22;

		double** M1;
		double** M2;
		double** M3;
		double** M4;
		double** M5;
		double** M6;
		double** M7;
		double** AResult;
		double** BResult;


		A11 = new double* [HalfSize];
		A12 = new double* [HalfSize];
		A21 = new double* [HalfSize];
		A22 = new double* [HalfSize];

		B11 = new double* [HalfSize];
		B12 = new double* [HalfSize];
		B21 = new double* [HalfSize];
		B22 = new double* [HalfSize];

		C11 = new double* [HalfSize];
		C12 = new double* [HalfSize];
		C21 = new double* [HalfSize];
		C22 = new double* [HalfSize];

		M1 = new double* [HalfSize];
		M2 = new double* [HalfSize];
		M3 = new double* [HalfSize];
		M4 = new double* [HalfSize];
		M5 = new double* [HalfSize];
		M6 = new double* [HalfSize];
		M7 = new double* [HalfSize];

		AResult = new double* [HalfSize];
		BResult = new double* [HalfSize];

#pragma omp parallel for
		for (int i = 0; i < HalfSize; i++)
		{
			A11[i] = new double[HalfSize];
			A12[i] = new double[HalfSize];
			A21[i] = new double[HalfSize];
			A22[i] = new double[HalfSize];

			B11[i] = new double[HalfSize];
			B12[i] = new double[HalfSize];
			B21[i] = new double[HalfSize];
			B22[i] = new double[HalfSize];

			C11[i] = new double[HalfSize];
			C12[i] = new double[HalfSize];
			C21[i] = new double[HalfSize];
			C22[i] = new double[HalfSize];

			M1[i] = new double[HalfSize];
			M2[i] = new double[HalfSize];
			M3[i] = new double[HalfSize];
			M4[i] = new double[HalfSize];
			M5[i] = new double[HalfSize];
			M6[i] = new double[HalfSize];
			M7[i] = new double[HalfSize];

			AResult[i] = new double[HalfSize];
			BResult[i] = new double[HalfSize];
		}
		/////////////////////////////////////////

		//Разделение матрицы на четыре части
#pragma omp parallel for
		for (int i = 0; i < HalfSize; i++)
		{
#pragma omp parallel for
			for (int j = 0; j < HalfSize; j++)
			{
				A11[i][j] = MatrixA[i][j];
				A12[i][j] = MatrixA[i][j + HalfSize];
				A21[i][j] = MatrixA[i + HalfSize][j];
				A22[i][j] = MatrixA[i + HalfSize][j + HalfSize];

				B11[i][j] = MatrixB[i][j];
				B12[i][j] = MatrixB[i][j + HalfSize];
				B21[i][j] = MatrixB[i + HalfSize][j];
				B22[i][j] = MatrixB[i + HalfSize][j + HalfSize];

			}
		}

		//P1 == M1[][]
		ADD(A11, A22, AResult, HalfSize);
		ADD(B11, B22, BResult, HalfSize);
		StrassenPar(AResult, BResult, M1, HalfSize, linearMultBlockSize);


		//P2 == M2[][]
		ADD(A21, A22, AResult, HalfSize);              //M2=(A21+A22)B11
		StrassenPar(AResult, B11, M2, HalfSize, linearMultBlockSize);       //Mul(AResult,B11,M2);

		//P3 == M3[][]
		SUB(B12, B22, BResult, HalfSize);              //M3=A11(B12-B22)
		StrassenPar(A11, BResult, M3, HalfSize, linearMultBlockSize);       //Mul(A11,BResult,M3);

		//P4 == M4[][]
		SUB(B21, B11, BResult, HalfSize);           //M4=A22(B21-B11)
		StrassenPar(A22, BResult, M4, HalfSize, linearMultBlockSize);       //Mul(A22,BResult,M4);

		//P5 == M5[][]
		ADD(A11, A12, AResult, HalfSize);           //M5=(A11+A12)B22
		StrassenPar(AResult, B22, M5, HalfSize, linearMultBlockSize);       //Mul(AResult,B22,M5);


		//P6 == M6[][]
		SUB(A21, A11, AResult, HalfSize);
		ADD(B11, B12, BResult, HalfSize);             //M6=(A21-A11)(B11+B12)
		StrassenPar(AResult, BResult, M6, HalfSize, linearMultBlockSize);    //Mul(AResult,BResult,M6);

		//P7 == M7[][]
		SUB(A12, A22, AResult, HalfSize);
		ADD(B21, B22, BResult, HalfSize);             //M7=(A12-A22)(B21+B22)
		StrassenPar(AResult, BResult, M7, HalfSize, linearMultBlockSize);     //Mul(AResult,BResult,M7);

		//C11 = M1 + M4 - M5 + M7;
		ADD(M1, M4, AResult, HalfSize);
		SUB(M7, M5, BResult, HalfSize);
		ADD(AResult, BResult, C11, HalfSize);

		//C12 = M3 + M5;
		ADD(M3, M5, C12, HalfSize);

		//C21 = M2 + M4;
		ADD(M2, M4, C21, HalfSize);

		//C22 = M1 + M3 - M2 + M6;
		ADD(M1, M3, AResult, HalfSize);
		SUB(M6, M2, BResult, HalfSize);
		ADD(AResult, BResult, C22, HalfSize);


		// Сбор частей в единую матрицу
#pragma omp parallel for
		for (int i = 0; i < HalfSize; i++)
		{
#pragma omp parallel for
			for (int j = 0; j < HalfSize; j++)
			{
				MatrixC[i][j] = C11[i][j];
				MatrixC[i][j + HalfSize] = C12[i][j];
				MatrixC[i + HalfSize][j] = C21[i][j];
				MatrixC[i + HalfSize][j + HalfSize] = C22[i][j];
			}
		}

		// очистка памяти
#pragma omp parallel for
		for (int i = 0; i < HalfSize; i++)
		{
			delete[] A11[i]; delete[] A12[i]; delete[] A21[i]; 	delete[] A22[i];
			delete[] B11[i]; delete[] B12[i]; delete[] B21[i];	delete[] B22[i];
			delete[] C11[i]; delete[] C12[i]; delete[] C21[i];	delete[] C22[i];
			delete[] M1[i]; delete[] M2[i]; delete[] M3[i]; delete[] M4[i];
			delete[] M5[i]; delete[] M6[i]; delete[] M7[i];
			delete[] AResult[i]; delete[] BResult[i];
		}
		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
		delete[] M1; delete[] M2; delete[] M3; delete[] M4; delete[] M5;
		delete[] M6; delete[] M7;
		delete[] AResult;
		delete[] BResult;


	}
	return 0;
}

int Validate_size(int size, int linearMultBlockSize)
{
	int tms = size;
	while (tms > linearMultBlockSize)
	{
		if (tms % 2 == 0)
		{
			tms /= 2;
		}
		else
			return 0;
	}
	return 1;
}

int Find_Valid_Size(int size, int linearMultBlockSize)
{
	int newsize = size;
	while (Validate_size(newsize, linearMultBlockSize) == 0)
	{
		newsize++;
	}
	return newsize;
}

double** ExpandMatrixToSize(double** matrix, int sizeA, int sizeB, int NewSize)
{
	double** NewMatr = new double* [NewSize];
	for (int i = 0; i < NewSize; i++)
	{
		NewMatr[i] = new double[NewSize];
		for (int j = 0; j < NewSize; j++)
		{
			NewMatr[i][j] = 0;
		}
	}
	for (int i = 0; i < sizeA; i++)
		for (int j = 0; j < sizeB; j++)
		{
			NewMatr[i][j] = matrix[i][j];
		}
	return NewMatr;
}

int Validate_sizePar(int size, int linearMultBlockSize)
{
	int tms = size;
	while (tms > linearMultBlockSize)
	{
		if (tms % 2 == 0)
		{
			tms /= 2;
		}
		else
			return 0;
	}
	return 1;
}

int Find_Valid_SizePar(int size, int linearMultBlockSize)
{
	int newsize = size;
	while (Validate_size(newsize, linearMultBlockSize) == 0)
	{
		newsize++;
	}
	return newsize;
}

double** ExpandMatrixToSizePar(double** matrix, int sizeA, int sizeB, int NewSize)
{
	double** NewMatr = new double* [NewSize];
#pragma omp parallel for
	for (int i = 0; i < NewSize; i++)
	{
		NewMatr[i] = new double[NewSize];
#pragma omp parallel for
		for (int j = 0; j < NewSize; j++)
		{
			NewMatr[i][j] = 0;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < sizeA; i++)
#pragma omp parallel for
		for (int j = 0; j < sizeB; j++)
		{
			NewMatr[i][j] = matrix[i][j];
		}
	return NewMatr;
}

double MulArr_Single_Fast(double** A, double** B, double** C, int sizeA, int sizeB, int sizeC, int linearMultBlockSize)
{
	double t_st = 0, t_ed = -1;
	int size = sizeA;
	double time = 0;

	if (sizeA == sizeB && sizeA == sizeC && Validate_size(size, linearMultBlockSize))
	{
		t_st = omp_get_wtime();
		Strassen(A, B, C, size, linearMultBlockSize);
		time = omp_get_wtime() - t_st;
	}
	else
	{
		//std::cout << "  Будет выполнена корректировка размера до квадрата - так чтобы до блока <=" << linearMultBlockSize << " выполнялось нормальное деление на 2" << std::endl;
		// Здесь код для перераспределения размера и повторного вызова Перемножения 
		if (size < sizeB) size = sizeB;
		if (size < sizeC) size = sizeC;
		if (size != Find_Valid_Size(size, linearMultBlockSize))
		{
			size = Find_Valid_Size(size, linearMultBlockSize);
		}
		//std::cout << "  Новый размер для перемножения методом Штрассена: " << size << "x" << size << std::endl;
		t_st = omp_get_wtime();
#pragma align(4)
		double** TA = ExpandMatrixToSize(A, sizeA, sizeB, size);
#pragma align(4)
		double** TB = ExpandMatrixToSize(B, sizeB, sizeC, size);
#pragma align(4)
		double** TC = ExpandMatrixToSize(C, 0, 0, size);
		t_ed = omp_get_wtime() - t_st;
		time = t_ed;
		//std::cout << "  Время затраченное на перераспределение памяти " << t_ed * 1000 << " ms." << std::endl;
		t_st = omp_get_wtime();
		Strassen(TA, TB, TC, size, linearMultBlockSize);
		t_ed = omp_get_wtime() - t_st;
		time += t_ed;
		//std::cout << "  Время затраченное на непосредственное перемножение " << t_ed * 1000 << " ms." << std::endl;
		t_st = omp_get_wtime();
		for (int i = 0; i < sizeA; i++)
			for (int j = 0; j < sizeC; j++)
			{
				C[i][j] = TC[i][j];
			}
		for (int i = 0; i < size; i++)
		{
			delete[] TA[i];
			delete[] TB[i];
			delete[] TC[i];
		}
		delete[] TA;
		delete[] TB;
		delete[] TC;
		t_ed = omp_get_wtime() - t_st;
		//std::cout << "  Время затраченное получение результата перемножения и очистку " << t_ed * 1000 << " ms." << std::endl;
		time += t_ed;
	}

	return time * 1000;
}

double MulArr_Par_Fast(double** A, double** B, double** C, int sizeA, int sizeB, int sizeC, int linearMultBlockSize)
{
	double t_st = 0, t_ed = -1;
	int size = sizeA;
	double time = 0;

	if (sizeA == sizeB && sizeA == sizeC && Validate_size(size, linearMultBlockSize))
	{
		t_st = omp_get_wtime();
		StrassenPar(A, B, C, size, linearMultBlockSize);
		time = omp_get_wtime() - t_st;
	}
	else
	{
		//std::cout << "  Будет выполнена корректировка размера до квадрата - так чтобы до блока <=" << linearMultBlockSize << " выполнялось нормальное деление на 2" << std::endl;
		// Здесь код для перераспределения размера и повторного вызова Перемножения 
		if (size < sizeB) size = sizeB;
		if (size < sizeC) size = sizeC;
		if (size != Find_Valid_SizePar(size, linearMultBlockSize))
		{
			size = Find_Valid_SizePar(size, linearMultBlockSize);
		}
		//std::cout << "  Новый размер для перемножения методом Штрассена: " << size << "x" << size << std::endl;
		t_st = omp_get_wtime();
#pragma align(4)
		double** TA = ExpandMatrixToSizePar(A, sizeA, sizeB, size);
#pragma align(4)
		double** TB = ExpandMatrixToSizePar(B, sizeB, sizeC, size);
#pragma align(4)
		double** TC = ExpandMatrixToSizePar(C, 0, 0, size);
		t_ed = omp_get_wtime() - t_st;
		time = t_ed;
		//std::cout << "  Время затраченное на перераспределение памяти " << t_ed * 1000 << " ms." << std::endl;
		t_st = omp_get_wtime();
		Strassen(TA, TB, TC, size, linearMultBlockSize);
		t_ed = omp_get_wtime() - t_st;
		time += t_ed;
		//std::cout << "  Время затраченное на непосредственное перемножение " << t_ed * 1000 << " ms." << std::endl;
		t_st = omp_get_wtime();
#pragma omp parallel for
		for (int i = 0; i < sizeA; i++)
#pragma omp parallel for
			for (int j = 0; j < sizeC; j++)
			{
				C[i][j] = TC[i][j];
			}
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			delete[] TA[i];
			delete[] TB[i];
			delete[] TC[i];
		}
		delete[] TA;
		delete[] TB;
		delete[] TC;
		t_ed = omp_get_wtime() - t_st;
		//std::cout << "  Время затраченное получение результата перемножения и очистку " << t_ed * 1000 << " ms." << std::endl;
		time += t_ed;

	}

	return time * 1000;
}

double TestIter(void* Funct, int iterations, double** A, double** B, double** C, int M, int N, int K, int linearMultBlockSize)
{
	
	double curtime = 0, avgTime = 0, avgTimeT = 0, correctAVG = 0;;
	double* Times = new double[iterations];
	for (int i = 0; i < iterations; i++)
	{
		std::cout << "*";
		// Запуск функции и получение времени в миллисекундах
		curtime = ((*(TestFunctTempl)Funct)(A, B, C, M, N, K, linearMultBlockSize)) * 1000;
		// запись времени в массив для определения среднеарифметического значения в доверительном интервале
		Times[i] = curtime;
		//       std::cout << curtime << std::endl;
	}
	//std::cout << "AvgTime:" << avgTime << std::endl;
	// Определения среднеарифметического значения в доверительном интервале по всем итерациям и вывод значения на экран
	avgTimeT = AvgTrustedInterval(avgTime, Times, iterations);
	//std::cout << "AvgTimeTrusted:" << avgTimeT << std::endl;
	return avgTimeT;
}

// Функция рассчета среднеарифметического значения в доверительном интервале
double AvgTrustedInterval(double& avg, double*& times, int& cnt)
{
	double sd = 0, newAVg = 0;
	int newCnt = 0;
	for (int i = 0; i < cnt; i++)
	{
		sd += (times[i] - avg) * (times[i] - avg);
	}
	sd /= (cnt - 1.0);
	sd = sqrt(sd);
	for (int i = 0; i < cnt; i++)
	{
		if (avg - sd <= times[i] && times[i] <= avg + sd)
		{
			newAVg += times[i];
			newCnt++;
		}
	}
	if (newCnt == 0) newCnt = 1;
	return newAVg / newCnt;
}
void test_functions(void** Functions, std::string(&function_names)[7], int iterations, double** A, double** B, double** C, int M, int N, int K, int linearMultBlockSize)
{
	double Functions_time_ms[7];
	// проведение замеров времени работы по каждой функции
	for (int i = 0; i < 7; i++)
	{
		Functions_time_ms[i] = TestIter(Functions[i], iterations, A, B, C, M, N, K, linearMultBlockSize);
		std::cout << std::endl;
	}

	std::ofstream out;          // поток для записи
	out.open("result.txt", std::ios::app); // окрываем файл для записи

	if (out.is_open())
		// Вывод результатов замера (можно организовать вывод в файл)
		for (int i = 0; i < 7; i++)
			out << function_names[i] << "\t\t\t" << Functions_time_ms[i] << " ms." << std::endl;

	out << std::endl;
	out.close();
}
int main()
{
	void** Functions = new void* [7]{ TestFillArr_Single, TestFillArr_Par, TestMulArr_Single, TestMulArr_Par1, TestMulArr_Par2, TestMulArr_Single_Fast, TestMulArr_Par_Fast };
	std::string  function_names[7]{ "TestFillArr_Single", "TestFillArr_Par", "TestMulArr_Single", "TestMulArr_Par1", "TestMulArr_Par2", "TestMulArr_Single_Fast", "TestMulArr_Par_Fast"};
	setlocale(LC_ALL, "Russian");

	/// Матрица A
	int			sizeA = 0,
				sizeB = 0,
				sizeC = 0;
	double**	A; // Массив
	double**	B; // Массив
	double**	C; // Массив


	std::ofstream out;
	out.open("result.txt", std::ios::app);
	if (out.is_open())
		out << "|||||-Замеры времени начаты: " << __TIMESTAMP__ << "-|||||" << std::endl;
	out.close();

	for (int i = 1; i < 5; i++)
	{
		out.open("result.txt", std::ios::app);
		if (out.is_open())
			out << "\t\tПОТОКОВ: " << i << std::endl << std::endl;
		out.close();

		std::cout << "\t\tПОТОКОВ: " << i << std::endl;
		omp_set_num_threads(i);

		for (long j = 400; j < 1600; j += 300)
		{
			out.open("result.txt", std::ios::app);
			if (out.is_open())
				out << "\tМАТРИЦЫ: " << j << " x " << j <<  std::endl << std::endl;
			out.close();

			sizeA = sizeB = sizeC = j;

			//выделение памяти для массива а, каждый элемент массива -
				//указатель на double
			A = new double* [sizeA];
			for (int i = 0; i < sizeA; i++)
				//выделение памяти для каждого элемента
				//a[i], a[i] адресует М элементов типа double
				A[i] = new double[sizeA];

			//выделение памяти для массива b, каждый элемент массива -
			//указатель на double
			B = new double* [sizeB];
			for (int i = 0; i < sizeB; i++)
				//выделение памяти для каждого элемента
				//b[i], b[i] адресует М элементов типа double
				B[i] = new double[sizeB];

			//выделение памяти для массива c, каждый элемент массива -
			//указатель на double
			C = new double* [sizeC];
			for (int i = 0; i < sizeC; i++)
				//выделение памяти для каждого элемента
				//c[i], c[i] адресует L элементов типа double
				C[i] = new double[sizeC];
			std::cout << "Набор данных:" << j << std::endl;
			test_functions(Functions, function_names, 20, A, B, C, sizeA, sizeB, sizeC, 64);

			for (int i = 0; i < sizeA; i++)
				//освобождение памяти для каждого элемента a[i]
				delete[] A[i];
			//освобождение памяти для а
			delete[] A;

			for (int i = 0; i < sizeB; i++)
				//освобождение памяти для каждого элемента b[i]
				delete[] B[i];
			//освобождение памяти для b
			delete[] B;

			for (int i = 0; i < sizeC; i++)
				//освобождение памяти для каждого элемента c[i]
				delete[] C[i];
			//освобождение памяти c
			delete[] C;
		}
	}
	std::cout << "Замеры времени выполнения функций выполнены, все данные записаны в файл result.txt." << std::endl;

	system("result.txt");
}