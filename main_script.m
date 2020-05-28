clear all
close all

%Download trained filters for reconstruction****
outfilename = websave('filters_ourVideos_obj2.45e+04filters_ourVideos_obj2.45e+04','https://www.dropbox.com/s/djtgw9ojxwtxj36/filters_ourVideos_obj2.45e%2B04.mat?dl=0');

%% Debug options
verbose = 'all';

%% Video Dataset
path_files = 'groundtruth_videos/';

%PSF
% load('psf.mat');
% K = K(:,:,1);
psf =zeros(5,5);
psf(3,3) = 1;


videoList = dir(strcat(path_files,'*.mat'));

for i = 1:length(videoList)
	%%CAPTURE SIMULATION
    load(strcat(path_files,videoList(i).name))
    T = 20;
    bump_size = 3;
    psize = [11 11 T];
    Mf = size(video,1);
    Nf = size(video,2);
    sampling3D = create_mask(Mf, Nf, T, psize,bump_size);
    outputfolder = strcat('results_20200310/',videoList(i).name(1:end-4));
    mkdir( outputfolder );
 
    close all
    coded_video = zeros(Mf,Nf);            
    for j =1:T
        sampling_i = sampling3D(:,:,j);
        transmit(j) = sum(sampling_i(:))/(size(sampling_i,1)*size(sampling_i,2));
        coded_video = coded_video + (video(:,:,j).*sampling3D(:,:,j));
    end
     coded_video_norm = (coded_video - min(coded_video(:))) / (max(coded_video(:))  - min(coded_video(:)));
     imwrite(coded_video_norm,strcat('results_20200310/',sprintf('%s_coded_bsize_%1d_T_%02d',videoList(i).name(1:end-4),bump_size,T),'.png'));
	
	%%RECONSTRUCTION
    for j=1:T-1
        signal_sparse = [];
        M = [sampling3D(:,:,j) sampling3D(:,:,j+1)];
        original = [video(:,:,j) video(:,:,j+1)];  
        signal_sparse_j = zeros(size(coded_video));
        M_j = sampling3D(:,:,j);
        signal_sparse_j(M_j == 1) = (1/3).*coded_video(M_j == 1);
        signal_sparse = [signal_sparse signal_sparse_j];
        signal_sparse_j = zeros(size(coded_video));
        M_j = sampling3D(:,:,j+1);
        signal_sparse_j(M_j == 1) = (1/3).*coded_video(M_j == 1);
        signal_sparse = [signal_sparse signal_sparse_j];

        %Initial simple interpolation
        aux = Interpolation_Initial(signal_sparse,~M);
        aux(aux>max(signal_sparse(:))) = max(signal_sparse(:));
        aux(aux<min(signal_sparse(:))) = min(signal_sparse(:));
        smooth_init = aux;

        %Local smooth
        k = fspecial('gaussian',[13 13],3*1.591); %Filter from local contrast normalization
        smooth_init = imfilter(smooth_init, k, 'same', 'conv', 'symmetric');


        %% 2) Reconstruct with our basis
        kernels = load('filters_ourVideos_obj2.45e+04.mat');
        d = kernels.d;

        %% 1) Sparse coding reconstruction     
        fprintf('Doing sparse coding reconstruction.\n\n')

        %Data term
        lambda_residual = 100.0;

        %Sparsity
        lambda = 10.0;

        %Temporal smoothness
        lambda_temp = 1;
        gamma_temp = 10;

        signal = original;           
        verbose_admm = 'all';
        max_it = [15]; 

        tic();        
        [ z, sig_rec_ours, sig_rec_ours_blurred] = admm_solve_conv2D_weighted_sampling(signal_sparse,[Mf Nf 2], d , M, psf, lambda_residual, lambda, lambda_temp,gamma_temp, smooth_init, max_it, 1e-5, signal, verbose_admm);
        tt = toc;
        save(sprintf('%s/%s_frame_%02d_bsize_%1d_L_%d_LS_%d_LT_%d_GT_%d_t_%03.1f.mat',outputfolder,videoList(i).name(1:end-4),j,bump_size,lambda_residual,lambda,lambda_temp,gamma_temp,tt),'sig_rec_ours', 'z');
        sig_rec_ours = (sig_rec_ours - min(sig_rec_ours(:))) / (max(sig_rec_ours(:)) - min(sig_rec_ours(:)));
        imwrite(sig_rec_ours,sprintf('%s/%s_frame_%02d_bsize_%1d_L_%d_LS_%d_LT_%d_GT_%d_t_%03.1f.png',outputfolder,videoList(i).name(1:end-4),j,bump_size,lambda_residual,lambda,lambda_temp,gamma_temp,tt))
    end

end

%%Reconstruction is done, now we take the output frames and put the video
%%together
disp('Reconstruction done! Now merging reconstructed frames into video')
clear all
close all

d = dir('results_20200310');
isub = [d(:).isdir]; 
aux = ~strcmp(struct2cell(d),'.') & ~strcmp(struct2cell(d),'..');
isub = isub & aux(1,:);
folders = {d(isub).name}';

for k=1:length(folders)
    fpath = strcat(folders{k},'/');
    list = dir(strcat('results_20200310/',fpath,'*.mat'));

    load(strcat('results_20200310/',fpath,list(1).name));
    M = size(sig_rec_ours,1);
    N = size(sig_rec_ours,2)/2;
    T = length(list);
    video = zeros(M, N, T);

    imgs = reshape(sig_rec_ours, M, N, []);
    video(:,:,1) = imgs(:,:,1);
    video(:,:,2) = imgs(:,:,2);
    for i=2:length(list)
        load(strcat('results_20200310/',fpath,list(i).name));
        imgs = reshape(sig_rec_ours, M, N, []);
        video(:,:,i) = (video(:,:,i) + imgs(:,:,1)) / 2;
        video(:,:,i+1) = imgs(:,:,2);
    end
    save(strcat('results_20200310/',folders{k},'_raw_video.mat'),'video')
    video = (video - min(video(:))) / (max(video(:)) - min(video(:)));
    v = VideoWriter(strcat('results_20200310/',folders{k},'_recon_video.mp4'),'MPEG-4');
    v.Quality = 100;
    v.FrameRate = 10;
    open(v);
    for t = 1:size(video,3)       
       frame = video(:,:,t);
       writeVideo(v,frame);
    end
    close(v);
    video = video.^(1/1.5);
    v = VideoWriter(strcat('results_20200310/',folders{k},'_recon_video_gamma1.5.mp4'),'MPEG-4');
    v.FrameRate = 10;
    v.Quality = 100;
    open(v);
    for t = 1:size(video,3)       
       frame = video(:,:,t);
       writeVideo(v,frame);
    end
    close(v);
end