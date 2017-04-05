clear all
close all
clc

REF = imread('hist_ref.jpg');
QUERY = imread('query_hist.jpg');
figure;imshow(REF);
figure;imshow(QUERY);
out = hist_match(REF,QUERY);
figure; imshow(out)