clear all;
close all;
clc
folder = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/Projects/Derain_knet_ECCV20/data/test/SOTS/indoor/SPAnet/norain/';
output_f = './rain800_out/';

listinfo = dir(strcat(folder,'*.png'));
lm = length(listinfo);
ps =[];
ss =[];
for i = 1:lm
   I = double(imread(strcat(folder,listinfo(i).name)));

   R = double(imread(strcat(output_f,listinfo(i).name)));
   [m,n,ch] = size(R);
   I = imresize(I,[m,n]);
%    R =255.0*( R)/( max(max(max(R))));
   ps=[ps psnr(R,I,255)];
   ss = [ss ssim(R,I)];
end