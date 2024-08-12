#!/bin/bash
make
python3 setup.py build_ext --inplace


python3 symnmf.py 4 sym inputs/spheres.txt > res/sym_spheres_res.txt
python3 symnmf.py 4 ddg inputs/spheres.txt > res/ddg_spheres_res.txt
python3 symnmf.py 4 norm inputs/spheres.txt > res/norm_spheres_res.txt
python3 symnmf.py 4 symnmf inputs/spheres.txt > res/symnmf_spheres_res.txt
python3 symnmf.py 3 sym inputs/iris.txt > res/sym_iris_res.txt
python3 symnmf.py 3 ddg inputs/iris.txt > res/ddg_iris_res.txt
python3 symnmf.py 3 norm inputs/iris.txt > res/norm_iris_res.txt
python3 symnmf.py 3 symnmf inputs/iris.txt > res/symnmf_iris_res.txt
python3 symnmf.py 3 sym inputs/drive1.txt > res/sym_drive1_res.txt
python3 symnmf.py 3 ddg inputs/drive1.txt > res/ddg_drive1_res.txt
python3 symnmf.py 3 norm inputs/drive1.txt > res/norm_drive1_res.txt
python3 symnmf.py 3 symnmf inputs/drive1.txt > res/symnmf_drive1_res.txt
python3 symnmf.py 3 sym inputs/drive2.txt > res/sym_drive2_res.txt
python3 symnmf.py 3 ddg inputs/drive2.txt > res/ddg_drive2_res.txt
python3 symnmf.py 3 norm inputs/drive2.txt > res/norm_drive2_res.txt
python3 symnmf.py 3 symnmf inputs/drive2.txt > res/symnmf_drive2_res.txt


./symnmf sym inputs/spheres.txt > res/symc_spheres_res.txt
./symnmf ddg inputs/spheres.txt > res/ddgc_spheres_res.txt
./symnmf norm inputs/spheres.txt > res/normc_spheres_res.txt
./symnmf sym inputs/iris.txt > res/symc_iris_res.txt
./symnmf ddg inputs/iris.txt > res/ddgc_iris_res.txt
./symnmf norm inputs/iris.txt > res/normc_iris_res.txt
./symnmf sym inputs/drive1.txt > res/symc_drive1_res.txt
./symnmf ddg inputs/drive1.txt > res/ddgc_drive1_res.txt
./symnmf norm inputs/drive1.txt > res/normc_drive1_res.txt
./symnmf sym inputs/drive2.txt > res/symc_drive2_res.txt
./symnmf ddg inputs/drive2.txt > res/ddgc_drive2_res.txt
./symnmf norm inputs/drive2.txt > res/normc_drive2_res.txt


python3 analysis.py 3 inputs/analysis/input_01.txt > res/analysis/input_01.txt
python3 analysis.py 3 inputs/analysis/input_02.txt > res/analysis/input_02.txt
python3 analysis.py 3 inputs/analysis/input_03.txt > res/analysis/input_03.txt
python3 analysis.py 6 inputs/analysis/input_04.txt > res/analysis/input_04.txt
python3 analysis.py 5 inputs/analysis/input_05.txt > res/analysis/input_05.txt
python3 analysis.py 5 inputs/analysis/input_06.txt > res/analysis/input_06.txt
python3 analysis.py 4 inputs/analysis/input_07.txt > res/analysis/input_07.txt
python3 analysis.py 8 inputs/analysis/input_08.txt > res/analysis/input_08.txt
python3 analysis.py 9 inputs/analysis/input_09.txt > res/analysis/input_09.txt
python3 analysis.py 2 inputs/analysis/input_10.txt > res/analysis/input_10.txt
python3 analysis.py 6 inputs/analysis/input_11.txt > res/analysis/input_11.txt
python3 analysis.py 5 inputs/analysis/input_12.txt > res/analysis/input_12.txt
python3 analysis.py 8 inputs/analysis/input_13.txt > res/analysis/input_13.txt
python3 analysis.py 3 inputs/analysis/iris.txt > res/analysis/iris.txt
python3 analysis.py 4 inputs/analysis/spheres.txt > res/analysis/spheres.txt


python3 compare.py
