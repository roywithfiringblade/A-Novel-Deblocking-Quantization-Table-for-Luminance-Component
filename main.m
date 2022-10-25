%%
% IVC Lab 2022 Final Optimization
% ===========================
% Last updated: 2022/08/17
% ##########Author##########
% name: Miao Zhang(03754260), Jiale Yu(03734976)
%% table setting
clear all;
close all;
L1 = [16,11,10,16,24,40,51,61; 12,12,14,19,26,58,60,55;...
    14,13,16,24,40,57,69,56; 14,17,22,29,51,87,80,62;...
    18,55,37,56,68,109,103,77; 24,35,55,64,81,104,113,92;...
    49,64,78,87,103,121,120,101; 72,92,95,98,112,100,103,99];
Uniform_table=ones(8);
Q_s=Q_star(8);
func=@MAPE;
x0 = [1];
res=patternsearch(func,x0,[],[],[],[],512/255,16);
Q_int=round(Q_s./res);
%% scales setting
s_scales=[0.15,0.3,0.7,1.0,1.5,3,5,7,10];
v_scales=[0.15,0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
%% default still dct-jpeg (chapter 4 result)

[s_R1,s_D1]=stillcodec(s_scales,L1);
%% default video dct-jpeg (chapter 5 result)

[v_R1,v_D1]=videocodec(v_scales,L1);
%% default still apbt-jpeg

[s_R2,s_D2]=stillcodec_apbt(s_scales,Uniform_table);
%% still dct-jpeg with Q* in eq(11) from the paper

[s_R3,s_D3]=stillcodec(s_scales,Q_s);
%% still dct-jpeg with Q_int in eq(14) from the paper

[s_R4,s_D4]=stillcodec(s_scales,Q_int);
%% video dct-jpeg with Q_int in eq(14) from the paper

[v_R2,v_D2]=videocodec(v_scales,Q_int);
%% plot RD-Curve results

figure(1)
plot(s_R2,s_D2,'b*-');
hold on
plot(s_R3,s_D3,'c*-');
hold on
legend('still apbt-jpeg ','still dct-jpeg with Q* ');
xlabel('bpp');
ylabel('PSNR[dB]');
xlim([0.2 4]);
%% plot RD-Curve results

figure(2)
plot(s_R1,s_D1,'r*-');
hold on
plot(s_R4,s_D4,'m*-');
hold on
plot(v_R1,v_D1,'ro-');
hold on
plot(v_R2,v_D2,'bo-');
legend('still default(chapter 4)','still dct-jpeg with Q^{int} ',...
    'video default(chapter 5)', 'video dct-jpeg with Q^{int} ');
xlabel('bpp');
ylabel('PSNR[dB]');
xlim([0.2 4]);

%% sub functions
function [R,D]=stillcodec(scales,L)
lena_small = double(imread('lena_small.tif'));
q=1;
for qScale = scales
   for d= 1:20  
        fore= ['./foreman20_40_RGB/foreman00',int2str(19+d),'.bmp'];
        fore=double(imread(fore));
        stillk  = IntraEncode(fore, qScale,L);       
        lenak  = IntraEncode(lena_small, qScale,L); 
        
        pmf = hist(lenak,-1000:4000);
        pmf = pmf/sum(pmf);
        [BinaryTree_s, HuffCode_s, BinCode_s, Codelengths_s] = buildHuffman(pmf);
        byte_still = enc_huffman(stillk+1001, BinCode_s, Codelengths_s);
        huffdecoded_still = dec_huffman(byte_still, BinaryTree_s, length(stillk)) -1001;
        n = (IntraDecode((huffdecoded_still),size(fore),qScale,L));  
        p_list(d) = calcPSNR(fore, ictYCbCr2RGB(n));
        b_list(d)=(numel(byte_still)*8) / (numel(fore)/3); 
   end
   R(q) = mean(b_list,'all');
   D(q) = mean(p_list,'all');
   fprintf('still scale: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, R(q), D(q))
q=q+1;
end
end

function [R,D]=videocodec(scales,L)
lena_small = double(imread('lena_small.tif'));
image1 = double(imread('./foreman20_40_RGB/foreman0020.bmp')); 
image2 = double(imread('./foreman20_40_RGB/foreman0021.bmp'));

q=1;
for qScale =  scales
k_small1=IntraEncode(lena_small, qScale,L); 
k= IntraEncode(image1, qScale,L);
pmf = hist(k_small1,-1000:4000);
pmf = pmf/sum(pmf);
[BinaryTree_1, HuffCode_1, BinCode_1, Codelengths_1] = buildHuffman(pmf);
bytestream_1 = enc_huffman(k+1001, BinCode_1, Codelengths_1);
k_rec = dec_huffman(bytestream_1, BinaryTree_1, length(k))-1001;
I_rec = (IntraDecode((k_rec), size(image1),qScale,L));
path = ['./foreman20_40_RGB/foreman_Q',num2str(qScale),'0020.bmp'];
imwrite(ictYCbCr2RGB(I_rec)/255,path);
Frame_cur_rec{1} = I_rec;

bpp(1) = (numel(bytestream_1)*8) / (numel(image1)/3);
PSNR(1) = calcPSNR(image1, ictYCbCr2RGB(I_rec));

Frame1_YCbCr = ictRGB2YCbCr(image1);
Frame2_YCbCr = ictRGB2YCbCr(image2);
m_vector = SSD(Frame1_YCbCr, Frame2_YCbCr);
pmf2 = hist(m_vector,1:81);
pmf2 = pmf2/sum(pmf2);
[BinaryTree_2, HuffCode_2,BinCode_2, Codelengths_2] = buildHuffman(pmf2);

for i = 1:20
    
    Frame_pre = ['./foreman20_40_RGB/foreman00',int2str(19+i),'.bmp'];
    Frame_cur = ['./foreman20_40_RGB/foreman00',int2str(19+i+1),'.bmp'];
    Frame_pre_YCbCr = ictRGB2YCbCr(double(imread(Frame_pre)));
    Frame_cur_YCbCr = ictRGB2YCbCr(double(imread(Frame_cur)));
        
    motion_vectors = SSD(Frame_cur_rec{i},Frame_cur_YCbCr);
    f_rec = SSD_rec(Frame_cur_rec{i},motion_vectors);
    
    bytestream_mv{i} = enc_huffman(motion_vectors(:), BinCode_2, Codelengths_2);
    mv_decode = dec_huffman(bytestream_mv{i}, BinaryTree_2, length(motion_vectors(:)));  
    mv_decode = reshape(mv_decode,size(motion_vectors));
    Frame_receive_rec = SSD_rec(Frame_cur_rec{i}, mv_decode);
    
    Prediction_error = ictYCbCr2RGB(Frame_cur_YCbCr - f_rec);
    err_en{i} = IntraEncode(Prediction_error, qScale,L);
    err_rec = IntraDecode(err_en{i}, size(Prediction_error) , qScale,L);
    Frame_cur_rec{i+1} = Frame_receive_rec + err_rec;
    path = ['./foreman20_40_RGB/foreman_Q',num2str(qScale),'00',int2str(19+i+1),'.bmp'];
    imwrite(ictYCbCr2RGB(Frame_cur_rec{i+1})/255,path);
    
    PSNR(i+1) = calcPSNR(ictYCbCr2RGB(Frame_cur_YCbCr),ictYCbCr2RGB(Frame_cur_rec{i+1})); 

end

Frame2_rec = SSD_rec(Frame1_YCbCr, m_vector);
Prediction_error1 = ictYCbCr2RGB(Frame2_rec - Frame2_YCbCr); 
Pre_error_encode = IntraEncode(Prediction_error1, qScale,L);
pmf3 = hist(Pre_error_encode,-1000:4000);
pmf3 = pmf3/sum(pmf3);
[BinaryTree_3, HuffCode_3, BinCode_3, Codelengths_3] = buildHuffman(pmf3);


for j=1:20
    bytestream_err{j} = enc_huffman(err_en{j}+1000+1, BinCode_3, Codelengths_3);
    bpp(j+1) = ((numel(bytestream_err{j})+numel(bytestream_mv{j}))*8) / (numel(Frame_pre_YCbCr)/3);
    err_rec = dec_huffman(bytestream_err{j}, BinaryTree_3, length(err_en{j})) -1000-1; 
end
D(q) = mean(PSNR,'all');
R(q) = mean(bpp,'all');

fprintf('video scale: %.2f bit-rate: %.2f bpp PSNR: %.2fdB\n', qScale, R(q),D(q));

q=q+1;
end
end
function [R,D]=stillcodec_apbt(scales,L)
lena_small = double(imread('lena_small.tif'));
q=1;
for qScale = scales
   for d= 1:20  
        fore= ['./foreman20_40_RGB/foreman00',int2str(19+d),'.bmp'];
        fore=double(imread(fore));
        stillk  = IntraEncode_apbt(fore, qScale,L);       
        lenak  = IntraEncode_apbt(lena_small, qScale,L); 
        
        pmf = hist(lenak,-1000:4000);
        pmf = pmf/sum(pmf);
        [BinaryTree_s, HuffCode_s, BinCode_s, Codelengths_s] = buildHuffman(pmf);
        byte_still = enc_huffman(stillk+1001, BinCode_s, Codelengths_s);
        huffdecoded_still = dec_huffman(byte_still, BinaryTree_s, length(stillk)) -1001;
        n = (IntraDecode_apbt((huffdecoded_still),size(fore),qScale,L));  
        p_list(d) = calcPSNR(fore, ictYCbCr2RGB(n));
        b_list(d)=(numel(byte_still)*8) / (numel(fore)/3);  
   end
   R(q) = mean(b_list,'all');
   D(q) = mean(p_list,'all');
   fprintf('still scale: %.2f bit-rate: %.2f bpp PSNR: %.2fdB\n', qScale, R(q), D(q))
q=q+1;
end
end
function dst = IntraEncode_apbt(image, qScale,L)
V=zeros(8);

for i=1:8
    for j=1:8
        if i==1
            V(i,j)=1/8;
        else
            V(i,j)=(8-i+sqrt(2))/64*cos((i-1)*(2*j-1)*pi/16);
        end           
    end
end
EoB=4000;
    imageYCbCr = ictRGB2YCbCr(image);
    I_dct=blockproc(imageYCbCr,[8,8],@(block_struct)APBT8x8(block_struct.data,V));
    I_quant=blockproc(I_dct,[8,8],@(block_struct)Quant8x8_apbt(block_struct.data, qScale,L));
    I_zz=blockproc(I_quant,[8,8],@(block_struct)ZigZag8x8(block_struct.data));
    h=size(I_zz,1);
    w=size(I_zz,2);
    I_zz=reshape(I_zz,[1,h*w]);
    dst=ZeroRunEnc_EoB(I_zz,EoB);
end
function dst = IntraDecode_apbt(image, sizeimg , qScale,L)
V_i=zeros(8);

for i=1:8
    for j=1:8
        if j==1
            V_i(i,j)=1;
        else
            V_i(i,j)=16/(8-j+sqrt(2))*cos((j-1)*(2*i-1)*pi/16);
        end           
    end
end
EoB=4000;
Id=ZeroRunDec_EoB(image,EoB);
l=sizeimg(1);
w=sizeimg(2);
c=sizeimg(3);
Id1=reshape(Id,[l*8,w/8*c]);
Id2=blockproc(Id1,[64,3],@(block_struct)DeZigZag8x8(block_struct.data));
I_dq=blockproc(Id2,[8,8],@(block_struct)DeQuant8x8_apbt(block_struct.data, qScale,L));
dst=blockproc(I_dq,[8,8],@(block_struct)IAPBT8x8(block_struct.data,V_i));


end
function dst = IntraEncode(image, qScale,L)
EoB=4000;
C=zeros(8);
for i=1:8
    for j=1:8
        if i==1
        C(i,j)=sqrt(1/8);
        else
            C(i,j)=cos((i-1)*(2*j-1)*pi/16)/2;
        end
    end
end
    imageYCbCr = ictRGB2YCbCr(image);
    I_dct=blockproc(imageYCbCr,[8,8],@(block_struct)DCT(block_struct.data,C));
    I_quant=blockproc(I_dct,[8,8],@(block_struct)Quant8x8(block_struct.data, qScale,L));
    I_zz=blockproc(I_quant,[8,8],@(block_struct)ZigZag8x8(block_struct.data));
    [h,w]=size(I_zz);
    I_zz=reshape(I_zz,[1,h*w]);
    dst=ZeroRunEnc_EoB(I_zz,EoB);
end
function dst = IntraDecode(image, sizeimg , qScale,L)
EoB=4000;
C=zeros(8);
for i=1:8
    for j=1:8
        if i==1
        C(i,j)=sqrt(1/8);
        else
            C(i,j)=cos((i-1)*(2*j-1)*pi/16)/2;
        end
    end
end
Id=ZeroRunDec_EoB(image,EoB);

l=sizeimg(1);
w=sizeimg(2);
c=sizeimg(3);
Id1=reshape(Id,[l*8,w/8*c]);
Id2=blockproc(Id1,[64,3],@(block_struct)DeZigZag8x8(block_struct.data));
I_dq=blockproc(Id2,[8,8],@(block_struct)DeQuant8x8(block_struct.data, qScale,L));
dst=blockproc(I_dq,[8,8],@(block_struct)IDCT(block_struct.data,C));

end

function loss=MAPE(lam)
Q_st=Q_star(8);
Q_s=Q_st./lam(1);
Q_int=round(Q_s);
r=Q_int./Q_s-1;
loss=sum(abs(r),'all')/64;
end
function Q=Q_star(N)
D_inv=zeros(N);
for i =1:N
if i==1
D_inv(i,i)=sqrt(N);
else
D_inv(i,i)=N*sqrt(2*N)/(sqrt(2)+N-i);
end
end
Q=zeros(N);
for i=1:N
for j=1:N
Q(i,j)=D_inv(i,i)*D_inv(j,j);
end
end
Q;
end
function mv_idx = SSD(ref_image, image)
[h,w,c] = size(image);
indexTable = reshape(1:81, 9, 9)';
posx = 1 : 8 : w;  
posy = 1 : 8 : h;
numx = w ./ 8;
numy = h ./ 8;
mv_idx = zeros(numy, numx);


for row = 1:numy
    for col = 1:numx
        locX = posx(col);  
        locY = posy(row);
        imgBlock = image(locY:locY+7, locX:locX+7);  
        flag = 0;
        for refX = locX-4:locX+4
            if (refX<1) | (refX>(w-7)) 
                continue
            end
            for refY = locY-4:locY+4
                if (refY<1) | (refY>(h-7)) 
                    continue
                end
                refBlock = ref_image(refY:refY+7, refX:refX+7);
                diff = (imgBlock - refBlock).^2;
                SSD = sum(diff(:));
                if ~flag
                    SSD_min = SSD;
                    flag = 1; 
                end
                if SSD <= SSD_min
                    SSD_min = SSD;
                    bestX = refX;
                    bestY = refY;
                end
            end
        end
        vector = [bestX, bestY] - [locX, locY] + 5;
        mv_idx(row, col) = indexTable(vector(2), vector(1));
    end
end
end
function rec_image = SSD_rec(ref_image, motion_vectors)

[h,w,c] = size(ref_image);
rec_image = zeros(h, w, 3);
numx = w ./ 8;
numy = h ./ 8;
indexTable = reshape(1:81, 9, 9)';
posx = 1 : 8 : w; 
posy = 1 : 8 : h;

for row = 1:numy
    for col = 1:numx
        index = motion_vectors(row, col);
        [vectorY, vectorX] = find(indexTable==index);  
        vectorX = vectorX - 5;  
        vectorY = vectorY - 5;
        locX = posx(col);  
        locY = posy(row);
        refX = locX + vectorX;  
        refY = locY + vectorY;
        block = ref_image(refY:refY+7, refX:refX+7, :);
        rec_image(locY:locY+7, locX:locX+7, :) = block;
    end
end

end
function coeff=DCT(block,C)
coeff = zeros(8, 8, 3);
for c=1:3 
    coeff(:, :, c) = C*block(:,:,c)*C';
end
end
function block=IDCT(coeff,C)
block = zeros(8, 8, 3);
for c=1:3 
    block(:, :, c) = C'*coeff(:,:,c)*C;
end
end
function coeff = APBT8x8(block,V)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
coeff = zeros(8, 8, 3);

for c=1:3
    coeff(:,:,c)=V*block(:,:,c)*V';
end
end
function block = IAPBT8x8(coeff,V_i)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
block = zeros(8, 8, 3);

for c=1:3
    block(:, :, c) = V_i*coeff(:, :, c)*V_i';
end
end
function quant = Quant8x8(dct_block, qScale,L)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
quant = zeros(8, 8, 3);
% C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
%     24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
%     99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
%     99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];
quant(:, :, 1) = round(dct_block(:, :, 1) ./ (L .* qScale));
quant(:, :, 2) = round(dct_block(:, :, 2) ./ (L .* qScale));
quant(:, :, 3) = round(dct_block(:, :, 3) ./ (L .* qScale));
end
function dct_block = DeQuant8x8(quant_block, qScale,L)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
dct_block = zeros(8, 8, 3);
% C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
%     24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
%     99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
%     99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];
dct_block(:, :, 1) = quant_block(:, :, 1) .* qScale .* L;
dct_block(:, :, 2) = quant_block(:, :, 2) .* qScale .* L;
dct_block(:, :, 3) = quant_block(:, :, 3) .* qScale .* L;
end
function quant = Quant8x8_apbt(dct_block, qScale,L)

quant = zeros(8, 8, 3);
quant(:, :, 1) = round(dct_block(:, :, 1) ./ (L .* qScale));
quant(:, :, 2) = round(dct_block(:, :, 2) ./ (L .* qScale));
quant(:, :, 3) = round(dct_block(:, :, 3) ./ (L .* qScale));
end
function dct_block = DeQuant8x8_apbt(quant_block, qScale,L)

dct_block = zeros(8, 8, 3);
dct_block(:, :, 1) = quant_block(:, :, 1) .* qScale .* L;
dct_block(:, :, 2) = quant_block(:, :, 2) .* qScale .* L;
dct_block(:, :, 3) = quant_block(:, :, 3) .* qScale .* L;
end
function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
%
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
zz = zeros(64, 3);
ZigZag = [1,2,6,7,15,16,28,29;...
    3,5,8,14,17,27,30,43;...
    4,9,13,18,26,31,42,44;...
    10,12,19,25,32,41,45,54;...
    11,20,24,33,40,46,53,55;...
    21,23,34,39,47,52,56,61;...
    22,35,38,48,51,57,60,62;...
    36,37,49,50,58,59,63,64];
for i=1:3
    quantBlock = quant(:, :, i);
    zz(ZigZag(:),  i) = quantBlock(:);
end
end
function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
coeffs = zeros(8, 8, 3);
ZigZag = [1,2,6,7,15,16,28,29;...
    3,5,8,14,17,27,30,43;...
    4,9,13,18,26,31,42,44;...
    10,12,19,25,32,41,45,54;...
    11,20,24,33,40,46,53,55;...
    21,23,34,39,47,52,56,61;...
    22,35,38,48,51,57,60,62;...
    36,37,49,50,58,59,63,64];
for i=1:3
    coeff = zz(ZigZag(:), i);
    coeffs(:, :, i) = reshape(coeff, 8, 8);
end
end
function dst = ZeroRunEnc_EoB(src,EoB)
[H,W]=size(src);
q=1;
m=W/64;
for j=1:m
    i=1;
    
    while i<=64
    if src(i+(j-1)*64)~=0
        dst(q)=src(i+(j-1)*64);
        i=i+1;
        q=q+1;
    else
        count=0;
        while (i+1)<=64 && src(i+1+(j-1)*64)==0
            count=count+1;
            i=i+1;
        end
        if i==64
            dst(q)=EoB;
        else
            q=q+1;
            dst(q)=count;
        end
        i=i+1;
        q=q+1;
    end
    end
    
end
end
function dst = ZeroRunDec_EoB(s,EoB)
[H,W]=size(s);
[i,j,q]=deal(1);

    while i<=W
        if s(i)~=0 && s(i)~=EoB
            dst(j)=s(i);
        else
            if s(i)==EoB
                dst(j:64*q)=0;
                j=64*q;
            else
                count=s(i+1);
                dst(j:j+count)=0;
                j=j+count;
                i=i+1;
            end
        end
            j=j+1;
            i=i+1;
            if j>(64*q)
                q=q+1;
            end
    end
end
function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
r=rgb(:,:,1);
g=rgb(:,:,2);
b=rgb(:,:,3);
y=0.299*r+0.587*g+0.114*b;
cb=-0.169*r-0.331*g+0.5*b;
cr=0.5*r-0.419*g-0.081*b;
yuv=cat(3,y,cb,cr);
end
function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE
y=yuv(:,:,1);
u=yuv(:,:,2);
v=yuv(:,:,3);
r=y+1.402*v;
g=y-0.344*u-0.714*v;
b=y+1.772*u;
rgb=cat(3,r,g,b);
end
function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
mse=calcMSE(Image, recImage);
PSNR=10*log10(255^2/mse);
end
function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
Image=double(Image);
recImage=double(recImage);
[w,h,c]=size(Image);
mse=sum((Image-recImage).*(Image-recImage),'all');
MSE=mse/(w*h*c);
end
function pmf = stats_marg_1D(sequence, range)
num = length(sequence); 
pmf = hist(sequence, range); 
pmf = pmf./num; 

end
function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p );

global y

p=p(:)/sum(p)+eps;              % normalize histogram
p1=p;                           % working copy

c=cell(length(p1),1);			% generate cell structure 

for i=1:length(p1)				% initialize structure
   c{i}=i;						
end

while size(c)-2					% build Huffman tree
	[p1,i]=sort(p1);			% Sort probabilities
	c=c(i);						% Reorder tree.
	c{2}={c{1},c{2}};           % merge branch 1 to 2
    c(1)=[];	                % omit 1
	p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2 
    p1(1)=[];	                % remove 1
end

%cell(length(p),1);              % generate cell structure
getcodes(c,[]);                  % recurse to find codes
code=char(y);

[numCodes maxlength] = size(code); % get maximum codeword length

% generate byte coded huffman table
% code

length_b=0;
HuffCode=zeros(1,numCodes);
for symbol=1:numCodes
    for bit=1:maxlength
        length_b=bit;
        if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
        elseif(code(symbol,bit)==char(48))
        else 
            length_b=bit-1;
            break;
        end;
    end;
    Codelengths(symbol)=length_b;
end;

BinaryTree = c;
BinCode = code;

clear global y;
end
function getcodes(a,dum)       
global y                            % in every level: use the same y
if isa(a,'cell')                    % if there are more branches...go on
         getcodes(a{1},[dum 0]);    % 
         getcodes(a{2},[dum 1]);
else   
   y{a}=char(48+dum);   
end
end
function [bytestream] = enc_huffman( data, BinCode,Codelengths)

a = BinCode(data(:),:)';
b = a(:);
mat = zeros(ceil(length(b)/8)*8,1);
p  = 1;
for i = 1:length(b)
    if b(i)~=' '
        mat(p,1) = b(i)-48;
        p = p+1;
    end
end
p = p-1;
mat = mat(1:ceil(p/8)*8);
d = reshape(mat,8,ceil(p/8))';
multi = [1 2 4 8 16 32 64 128];
bytestream = sum(d.*repmat(multi,size(d,1),1),2);
end
function [output] = dec_huffman (bytestream, BinaryTree, nr_symbols)

output = zeros(1,nr_symbols);
ctemp = BinaryTree;

dec = zeros(size(bytestream,1),8);
for i = 8:-1:1
    dec(:,i) = rem(bytestream,2);
    bytestream = floor(bytestream/2);
end

dec = dec(:,end:-1:1)';
a = dec(:);

i = 1;
p = 1;
while(i <= nr_symbols)&&p<=max(size(a))
    while(isa(ctemp,'cell'))
        next = a(p)+1;
        p = p+1;
        ctemp = ctemp{next};
    end
    output(i) = ctemp;
    ctemp = BinaryTree;
    i=i+1;
end
end