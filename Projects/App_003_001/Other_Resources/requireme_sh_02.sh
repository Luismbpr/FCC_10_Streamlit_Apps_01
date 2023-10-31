#!bin/bash
# Ask the user for app name or number
echo App name:
read app_name
pip list > pip_list_$app_name.txt
conda list > conda_list_$app_name.txt
conda list export > conda_list_export_$app_name.txt
conda freeze > conda_freeze_$app_name.txt
pip freeze > pip_freeze_$app_name.txt
python --version > python_version_$app_name.txt

echo " " >> requirements_$app_name.txt
echo " " >> requirements_$app_name.txt

echo "pip list start" >> requirements_$app_name.txt
pip list >> requirements_$app_name.txt
echo "pip list end" >> requirements_$app_name.txt

echo " " >> requirements_$app_name.txt
echo " " >> requirements_$app_name.txt

echo "conda list start" >> requirements_$app_name.txt
conda list >> requirements_$app_name.txt
echo "conda list end" >> requirements_$app_name.txt

echo " " >> requirements_$app_name.txt
echo " " >> requirements_$app_name.txt

echo "conda list start" >> requirements_$app_name.txt
conda list export >> requirements_$app_name.txt
echo "conda list end" >> requirements_$app_name.txt

echo " " >> requirements_$app_name.txt
echo " " >> requirements_$app_name.txt

echo "conda freeze start" >> requirements_$app_name.txt
conda freeze >> requirements_$app_name.txt
echo "conda freeze end" >> requirements_$app_name.txt

echo " " >> requirements_$app_name.txt
echo " " >> requirements_$app_name.txt

echo "pip freeze start" >> requirements_$app_name.txt
pip freeze >> requirements_$app_name.txt
echo "pip freeze end" >> requirements_$app_name.txt

echo " " >> requirements_$app_name.txt
echo " " >> requirements_$app_name.txt