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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>


/* Reference: https://brc2.com/the-algorithm-workshop/ */
/* Calculate assignment to minimize cost */

template<typename T>
class HungarianAlgorithm
{
public:
    HungarianAlgorithm(const std::vector<std::vector<T>>& cost_matrix)
    {
        rows = static_cast<int32_t>(cost_matrix.size());
        cols = static_cast<int32_t>(cost_matrix[0].size());
        C = cost_matrix;
        M = std::vector<std::vector<int32_t>>(rows, std::vector<int32_t>(cols));
        row_cover = std::vector<int32_t>(rows, 0);
        col_cover = std::vector<int32_t>(cols, 0);

        path = std::vector<std::vector<int32_t>>(rows * cols, std::vector<int32_t>(2));
        path_count = 0;
        path_row_0 = 0;
        path_col_0 = 0;
    }

    ~HungarianAlgorithm() {}

    void Solve(std::vector<int32_t>& assign_for_row, std::vector<int32_t>& assign_for_col)
    {
        int32_t step = 1;
        for (bool done = false; !done;) {
            //printf("\n === step: %d\n", step);
            switch (step) {
            case 1:
                step = Step1();
                break;
            case 2:
                step = Step2();
                break;
            case 3:
                step = Step3();
                break;
            case 4:
                step = Step4();
                break;
            case 5:
                step = Step5();
                break;
            case 6:
                step = Step6();
                break;
            default:
                done = true;
                break;
            }

#if 0
            DisplayCostMatrix();
            DisplayMaskMatrix();
            printf("\nrow_cover\n");
            for (int32_t i = 0; i < row_cover.size(); i++) {
                printf("%d  ", row_cover[i]);
            }
            printf("\ncol_cover\n");
            for (int32_t i = 0; i < col_cover.size(); i++) {
                printf("%d  ", col_cover[i]);
            }
            printf("\n");
#endif
        }

        for (int32_t y = 0; y < rows; y++) {
            for (int32_t x = 0; x < cols; x++) {
                if (M[y][x]) {
                    assign_for_row[y] = x;
                    assign_for_col[x] = y;
                }
            }
        }
    }




private:
    void DisplayCostMatrix()
    {
        printf("\nCost matrix\n");
        for (int32_t y = 0; y < C.size(); y++) {
            for (int32_t x = 0; x < C[0].size(); x++) {
                printf("%4.1f  ", C[y][x]);
            }
            printf("%\n");
        }
    }

    void DisplayMaskMatrix()
    {
        printf("\nMask matrix\n");
        for (int32_t y = 0; y < M.size(); y++) {
            for (int32_t x = 0; x < M[0].size(); x++) {
                printf("%d  ", M[y][x]);
            }
            printf("%\n");
        }
    }

    int32_t Step1()
    {
        for (int32_t y = 0; y < rows; y++) {
            T min_in_row = C[y][0];
            for (int32_t x = 1; x < cols; x++) {
                min_in_row = (std::min)(min_in_row, C[y][x]);
            }
            for (int32_t x = 0; x < cols; x++) {
                C[y][x] -= min_in_row;
            }
        }
        return 2;
    }

    int32_t Step2()
    {
        for (int32_t y = 0; y < rows; y++) {
            for (int32_t x = 0; x < cols; x++) {
                if (C[y][x] == 0 && row_cover[y] == 0 && col_cover[x] == 0) {
                    M[y][x] = 1;
                    row_cover[y] = 1;
                    col_cover[x] = 1;
                }
            }
        }
        for (int32_t y = 0; y < rows; y++) {
            row_cover[y] = 0;
        }
        for (int32_t x = 0; x < cols; x++) {
            col_cover[x] = 0;
        }
        return 3;
    }

    int32_t Step3()
    {
        for (int32_t y = 0; y < rows; y++) {
            for (int32_t x = 0; x < cols; x++) {
                if (M[y][x] == 1) {
                    col_cover[x] = 1;
                }
            }
        }
        int32_t col_count = 0;
        for (int32_t x = 0; x < cols; x++) {
            if (col_cover[x] == 1) {
                col_count++;
            }
        }
        if (col_count >= cols || col_count >= rows) {
            return 7;
        } else {
            return 4;
        }
    }

    int32_t Step4()
    {
        int32_t y = -1;
        int32_t x = -1;
        for (;;) {
            FindZero(y, x);
            if (y == -1) {
                return 6;
            } else {
                M[y][x] = 2;
                if (StarInRow(y)) {
                    x = FindStarInRow(y);
                    row_cover[y] = 1;
                    col_cover[x] = 0;
                } else {
                    path_row_0 = y;
                    path_col_0 = x;
                    return 5;
                }
            }
        }
    }

    int32_t Step5()
    {
        bool done = false;
        int32_t y = -1;
        int32_t x = -1;
        path_count = 1;
        path[path_count - 1][0] = path_row_0;
        path[path_count - 1][1] = path_col_0;
        while (!done) {
            y = FindStarInCol(path[path_count - 1][1]);
            if (y > -1) {
                path_count++;
                path[path_count - 1][0] = y;
                path[path_count - 1][1] = path[path_count - 2][1];
            } else {
                done = true;
            }

            if (!done) {
                x = FindPrimeInRow(path[path_count - 1][0]);
                path_count++;
                path[path_count - 1][0] = path[path_count - 2][0];
                path[path_count - 1][1] = x;
            }
        }
        AugmentPath();
        ClearCovers();
        ErasePrimes();
        return 3;

    }


    int32_t Step6()
    {
        T minval = FindSmallest();
        for (int32_t r = 0; r < rows; r++) {
            for (int32_t c = 0; c < cols; c++) {
                if (row_cover[r] == 1) {
                    C[r][c] += minval;
                }
                if (col_cover[c] == 0) {
                    C[r][c] -= minval;
                }
            }
        }
        return 4;
    }

    void FindZero(int32_t& y_of_zero, int32_t& x_of_zero)
    {
        y_of_zero = -1;
        x_of_zero = -1;
        for (int32_t y = 0; y < rows; y++) {
            for (int32_t x = 0, done = 0; x < cols && done == 0; x++) {
                if (C[y][x] == 0 && row_cover[y] == 0 && col_cover[x] == 0) {
                    y_of_zero = y;
                    x_of_zero = x;
                    done = 1;
                }
            }
        }
    }

    bool StarInRow(int32_t y)
    {
        bool temp = false;
        for (int32_t x = 0; x < cols; x++) {
            if (M[y][x] == 1) {
                temp = true;
            }
        }
        return temp;
    }

    int32_t FindStarInRow(int32_t y)
    {
        for (int32_t x = 0; x < cols; x++) {
            if (M[y][x] == 1) {
                return x;
            }
        }
        return -1;
    }

    int32_t FindStarInCol(int32_t x)
    {
        for (int32_t y = 0; y < rows; y++) {
            if (M[y][x] == 1) {
                return y;
            }
        }
        return -1;
    }

    int32_t FindPrimeInRow(int32_t y)
    {
        for (int32_t x = 0; x < cols; x++) {
            if (M[y][x] == 2) {
                return x;
            }
        }
        return -1;
    }

    void AugmentPath()
    {
        for (int32_t p = 0; p < path_count; p++) {
            if (M[path[p][0]][path[p][1]] == 1) {
                M[path[p][0]][path[p][1]] = 0;
            } else {
                M[path[p][0]][path[p][1]] = 1;
            }
        }
    }

    void ClearCovers()
    {
        for (int32_t y = 0; y < rows; y++) {
            row_cover[y] = 0;
        }
        for (int32_t x = 0; x < cols; x++) {
            col_cover[x] = 0;
        }
    }

    void ErasePrimes()
    {
        for (int32_t y = 0; y < rows; y++) {
            for (int32_t x = 0; x < cols; x++) {
                if (M[y][x] == 2) {
                    M[y][x] = 0;
                }
            }
        }
    }

    T FindSmallest()
    {
        T minval = std::numeric_limits<T>::infinity();
        for (int32_t y = 0; y < rows; y++) {
            for (int32_t x = 0; x < cols; x++) {
                if (row_cover[y] == 0 && col_cover[x] == 0) {
                    minval = (std::min)(minval, C[y][x]);
                }
            }
        }
        return minval;
    }

private:
    int32_t rows;
    int32_t cols;
    std::vector<std::vector<T>> C;          /* cost matrix */
    std::vector<std::vector<int32_t>> M;    /* mask matrix (1: starred zero, 2: primed zero) */
    std::vector<int32_t> row_cover;
    std::vector<int32_t> col_cover;
    std::vector<std::vector<int32_t>> path;
    int32_t path_count;
    int32_t path_row_0;
    int32_t path_col_0;
};
