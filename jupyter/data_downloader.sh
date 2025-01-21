#!/bin/bash

output_folder="data"
mkdir -p $output_folder

datasets=('1_ALOI' '2_annthyroid' '3_backdoor' '4_breastw' '5_campaign' '6_cardio' '7_Cardiotocography' 
'8_celeba' '9_census' '10_cover' '11_donors' '12_fault' '13_fraud' '14_glass' '15_Hepatitis'
'16_http' '17_InternetAds' '18_Ionosphere' '19_landsat' '20_letter' '21_Lymphography' 
'22_magic.gamma' '23_mammography' '24_mnist' '25_musk' '26_optdigits' '27_PageBlocks'
'28_pendigits' '29_Pima' '30_satellite' '31_satimage-2' '32_shuttle' '33_skin' 
'34_smtp' '35_SpamBase' '36_speech' '37_Stamps' '38_thyroid' '39_vertebral' 
'40_vowels' '41_Waveform' '42_WBC' '43_WDBC' '44_Wilt' '45_wine' '46_WPBC' '47_yeast')

base_url="https://github.com/Minqi824/ADBench/raw/refs/heads/main/adbench/datasets/Classical"

# loop through each dataset and download the file
for dataset in "${datasets[@]}"; do
    echo "Downloading dataset: $dataset"
    wget -P "$output_folder" "$base_url/$dataset.npz"
done
